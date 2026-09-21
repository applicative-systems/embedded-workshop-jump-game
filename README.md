# Nixcon 2026 workshop: From zero to product & computer vision with NVIDIA CUDA on a Jetson nano

This repository contains part 1 of our Nixcon 2026 workshop.
It shows how to deploy a toy computer vision game on an NVIDIA Jetson Orin Nano.

NixOS makes it easy to get toolchain dependencies and configuration fine tuning right.

## Contents

| Path         | What                                                                                 |
| ------------ | ------------------------------------------------------------------------------------ |
| `flake.nix`  | Wires up packages, checks, the `nixosConfiguration`s                                 |
| `jump-game/` | Example src. Completely vibe-coded, as workshop is about NixOS, not Python           |
| `modules/`   | Composable NixOS modules for HW support, app setup, size reduction, boot speed, etc. |
| `hosts/`     | Deployment-specific bits (NVMe install via disko, USB image)                         |

## NixOS configs

The `nixosConfiguration` attributes in the flake are nearly identical and only
differ in their disk deployment strategy: USB stick or NVMe disk.

- **`nano-usb`** is for raw disk images. Build with:

  ```bash
  nix build .#packages.aarch64-linux.diskImage
  ```

- **`nano`** is installed directly onto an NVMe drive on running device.
  Use a USB boot or try nixos-anywhere
  (kexec did not work for us with neither the Ubuntu image nor NixOS usb stick).

## Updating the firmware with NVIDIA's Jetson ISO

The r39 firmware is available as a **UEFI capsule update**, applied by NVIDIA's
[Jetson ISO installer](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)
Super easy: Put on USB stick, boot, abort Ubuntu installation.
Gives recent firmware.
From there, use `hardware.nvidia-jetpack.firmware.autoUpdate`.

### Cross-compilation

Cross-compiling a Jetson NixOS system from x86_64 works.
The catch is CUDA.
nixpkgs cannot cross-compile CUDA packages:

```
error: cudaPackages_13_2.backendStdenv has failed assertions:
- Requested Jetson CUDA capabilities (["8.7"]) require hostPlatform
  (x86_64-linux) to be aarch64-linux
```

The demo uses CUDA, so we dropped all cross compilation in this repo.
