// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
// Real DXR 1.1 inline ray queries, including a second shadow ray.
RaytracingAccelerationStructure scene : register(t0);
StructuredBuffer<float3> vertices : register(t1);
RWTexture2D<float4> color : register(u0);
RWTexture2D<float2> motion : register(u1);
RWTexture2D<float> depth : register(u2);
cbuffer Frame : register(b0) { uint width; uint height; uint padding0; uint padding1; }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.x >= width || tid.y >= height) return;
  float2 uv = (float2(tid.xy) + 0.5) / float2(width, height) * 2 - 1;
  uv.x *= float(width) / height;
  float3 origin = float3(0, 1.35, -4.5);
  float3 direction = normalize(float3(uv.x * 0.58, -uv.y * 0.58 - 0.13, 1));
  RayDesc ray = {origin, 0.01, direction, 100.0};
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
  while (query.Proceed()) {}
  float3 radiance = lerp(float3(0.055, 0.09, 0.15), float3(0.3, 0.5, 0.8), saturate(direction.y));
  float z = 1;
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    uint primitive = query.CommittedPrimitiveIndex();
    float3 a = vertices[primitive * 3], b = vertices[primitive * 3 + 1], c = vertices[primitive * 3 + 2];
    float3 normal = normalize(cross(b - a, c - a));
    if (dot(normal, direction) > 0) normal = -normal;
    float3 position = origin + query.CommittedRayT() * direction;
    float3 light = normalize(float3(-0.65, 1.3, -0.8));
    RayDesc shadow = {position + normal * 0.003, 0.001, light, 50.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> occlusion;
    occlusion.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, shadow);
    while (occlusion.Proceed()) {}
    float visibility = occlusion.CommittedStatus() == COMMITTED_NOTHING ? 1 : 0;
    float3 albedo = primitive < 2 ? float3(0.4, 0.43, 0.5) : float3(0.08, 0.65, 0.38);
    if (primitive < 2) albedo *= 0.65 + 0.35 * ((int(floor(position.x)) + int(floor(position.z))) & 1);
    float diffuse = max(dot(normal, light), 0) * visibility;
    float specular = pow(max(dot(normal, normalize(light - direction)), 0), 48) * visibility;
    radiance = albedo * (0.16 + 1.5 * diffuse) + 0.7 * specular;
    // Standard non-inverted perspective depth, near=.1, far=100.
    float viewZ = max(position.z - origin.z, 0.1);
    z = saturate(100.0 / 99.9 - 10.0 / (99.9 * viewZ));
  }
  color[tid.xy] = float4(radiance, 1);
  motion[tid.xy] = 0; // Fixed camera + static geometry: exact zero motion.
  depth[tid.xy] = z;
}
