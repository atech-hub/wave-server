//! Self-contained GPU accelerator for inference.
//!
//! Provides GPU-accelerated matrix-vector multiply for the forward pass.
//! No dependency on kerr-engine. Compiles the matvec shader at startup,
//! pre-allocates buffers, dispatches via wgpu. The Kerr-ODE, layer norm,
//! and attention scores stay on CPU (not the bottleneck).

use std::collections::HashMap;

use wgpu;
use bytemuck;
use pollster;

// ─── Dims uniform for shader ─────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    out_dim: u32,
    in_dim: u32,
}

// ─── Weight buffer cache key ─────────────────────────────────

#[derive(Hash, Eq, PartialEq)]
struct BufferKey {
    out_dim: usize,
    in_dim: usize,
    ptr: usize, // raw pointer as identity — same weight matrix = same buffer
}

// ─── GPU Accelerator ─────────────────────────────────────────

pub struct GpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Cached weight+bias GPU buffers (uploaded once per unique matrix)
    weight_cache: HashMap<BufferKey, (wgpu::Buffer, wgpu::Buffer)>,
    /// Pre-allocated scratch buffers for input/output
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    dims_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Max dimension these scratch buffers support
    max_dim: usize,
    pub adapter_name: String,
    pub backend_name: String,
}

impl GpuAccelerator {
    /// Create a GPU accelerator. Panics if no GPU available.
    pub fn new(max_dim: usize) -> Self {
        Self::with_device(max_dim, None)
    }

    /// Create with a specific GPU device index.
    pub fn with_device(max_dim: usize, device_idx: Option<usize>) -> Self {
        let instance = wgpu::Instance::default();
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());

        let adapter = if let Some(idx) = device_idx {
            adapters.into_iter().nth(idx)
                .expect(&format!("GPU device {idx} not found"))
        } else {
            adapters.into_iter().next()
                .expect("No GPU adapter found")
        };

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        println!("  GPU: {} ({})", adapter_name, backend_name);

        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wave-server"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
        ).expect("Failed to request GPU device");

        // Compile matvec shader
        let shader_src = include_str!("../shaders/matvec.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_layout"),
            entries: &[
                // dims uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // weight storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // bias storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // input storage
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output storage (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pre-allocate scratch buffers for max dimension
        let max_bytes = (max_dim * 4) as u64;

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: max_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: max_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims"),
            size: 8, // 2 × u32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: max_bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            weight_cache: HashMap::new(),
            input_buffer,
            output_buffer,
            dims_buffer,
            staging_buffer,
            max_dim,
            adapter_name,
            backend_name,
        }
    }

    /// GPU matrix-vector multiply: y = W @ x + b
    pub fn linear(&mut self, w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let in_dim = if out_dim > 0 { w[0].len() } else { 0 };
        assert!(out_dim <= self.max_dim && in_dim <= self.max_dim,
            "Dimension {out_dim}x{in_dim} exceeds GPU max_dim {}", self.max_dim);

        // Upload weight+bias (cached by pointer identity)
        let key = BufferKey { out_dim, in_dim, ptr: w.as_ptr() as usize };
        if !self.weight_cache.contains_key(&key) {
            let flat_w: Vec<f32> = w.iter().flat_map(|row| row.iter().copied()).collect();
            let w_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (flat_w.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&w_buf, 0, bytemuck::cast_slice(&flat_w));

            let b_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (b.len() * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(b));

            self.weight_cache.insert(key, (w_buf, b_buf));
        }
        let key = BufferKey { out_dim, in_dim, ptr: w.as_ptr() as usize };
        let (w_buf, b_buf) = self.weight_cache.get(&key).unwrap();

        // Upload dims + input
        let dims = Dims { out_dim: out_dim as u32, in_dim: in_dim as u32 };
        self.queue.write_buffer(&self.dims_buffer, 0, bytemuck::bytes_of(&dims));
        self.queue.write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(x));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.output_buffer.as_entire_binding() },
            ],
        });

        // Dispatch
        let workgroups = (out_dim as u32 + 63) / 64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging
        let out_bytes = (out_dim * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = self.staging_buffer.slice(..out_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        result[..out_dim].to_vec()
    }

    /// GPU matrix-vector multiply without bias: y = W @ x
    pub fn linear_no_bias(&mut self, w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let out_dim = w.len();
        let zero_bias = vec![0.0f32; out_dim];
        self.linear(w, &zero_bias, x)
    }
}

// GpuAccelerator is Send+Sync — wgpu Device and Queue are thread-safe
unsafe impl Send for GpuAccelerator {}
unsafe impl Sync for GpuAccelerator {}
