// The browser supplies optix.h and declares params from the JS descriptors.
// This is compiled by NVRTC and OptiX, not translated into WGSL.
extern "C" __global__ void __raygen__main() {
  const uint3 index = optixGetLaunchIndex();
  const uint3 size = optixGetLaunchDimensions();
  const float x = 2.f * (float(index.x) + 0.5f) / float(size.x) - 1.f;
  const float y = 2.f * (float(index.y) + 0.5f) / float(size.y) - 1.f;
  unsigned hit = 0, secondary = 0;
  optixTrace(params.scene, make_float3(x,y,-2.f), make_float3(0,0,1),
             0.001f, 1000.f, 0.f, 255, OPTIX_RAY_FLAG_NONE, 0, 1, 0,
             hit, secondary);
  const unsigned i = index.y * size.x + index.x;
  params.hits[i] = hit + params.offset;
  const uchar4 color = hit ? make_uchar4((unsigned char)params.tint,50,10,255)
                          : make_uchar4(10,30,180,255);
  params.pixels[i] = unsigned(color.x) | (unsigned(color.y)<<8) |
                     (unsigned(color.z)<<16) | (255u<<24);
  params.hdr[i] = make_float4(float(color.x) / 42.5f, float(color.y) / 42.5f,
                              float(color.z) / 42.5f, 1.f);
  surf2Dwrite(color, params.target, index.x * 4, index.y);
}
extern "C" __global__ void __miss__main() {
  optixSetPayload_0(0);
}
extern "C" __global__ void __anyhit__main() {
  if (params.tag == 0) optixIgnoreIntersection();
}
extern "C" __global__ void __closesthit__main() {
  // A reflected secondary ray is issued entirely from this CUDA hit program.
  // The plane faces the camera, so its reflection sees the miss program.
  const float3 origin = optixGetWorldRayOrigin();
  const float3 direction = optixGetWorldRayDirection();
  const float distance = optixGetRayTmax();
  unsigned reflected = 0, secondary = 1;
  if (optixGetPayload_1() == 0) {
    optixTrace(params.scene, make_float3(origin.x,origin.y,origin.z+distance-0.001f),
               make_float3(direction.x,direction.y,-direction.z),
               0.0001f,1000.f,0.f,255,OPTIX_RAY_FLAG_NONE,0,1,0,
               reflected,secondary);
  }
  optixSetPayload_0(params.tag + reflected);
}
