{ modulesPath, ... }:
{
  imports = [
    "${modulesPath}/virtualisation/disk-image.nix"
  ];
  image.format = "raw";
}
