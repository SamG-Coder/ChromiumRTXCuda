# ChromiumRTXCuda

![ChromiumRTXCuda icon](rtx_cuda/assets/icon-128.png)

Windows development fork of Chromium with native CUDA and RTX access behind a
normal website permission. The JavaScript API follows
[cuda-webshader](https://github.com/SamG-Coder/cuda-webshader)'s `GpuRuntime` design.
Public NVIDIA DLSS Super Resolution and DLAA are included; DLSS 5 is excluded.

See [the API, build instructions, demo, and current limitations](rtx_cuda/README.md).
This is an experimental native GPU integration for trusted development sites;
the GPU helper is not yet a hardened browser sandbox.

## Upstream Chromium

![Logo](chrome/app/theme/chromium/product_logo_64.png)

Chromium is an open-source browser project that aims to build a safer, faster,
and more stable way for all users to experience the web.

The project's web site is https://www.chromium.org.

To check out the source code locally, don't use `git clone`! Instead,
follow [the instructions on how to get the code](docs/get_the_code.md).

Documentation in the source is rooted in [docs/README.md](docs/README.md).

Learn how to [Get Around the Chromium Source Code Directory
Structure](https://www.chromium.org/developers/how-tos/getting-around-the-chrome-source-code).

For historical reasons, there are some small top level directories. Now the
guidance is that new top level directories are for product (e.g. Chrome,
Android WebView, Ash). Even if these products have multiple executables, the
code should be in subdirectories of the product.

If you found a bug, please file it at https://crbug.com/new.
