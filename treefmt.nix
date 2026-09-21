{
  lib,
  treefmt,
  deadnix,
  nixfmt,
  prettier,
  shellcheck,
  shfmt,
  statix,
}:

treefmt.withConfig {
  settings = {
    tree-root-file = "flake.nix";
    on-unmatched = "info";
    formatter = {
      nixfmt = {
        command = lib.getExe nixfmt;
        includes = [ "*.nix" ];
      };
      statix = {
        command = lib.getExe statix;
        options = [ "fix" ];
        no-positional-arg-support = true;
        includes = [ "*.nix" ];
      };
      deadnix = {
        command = lib.getExe deadnix;
        options = [ "--edit" ];
        includes = [ "*.nix" ];
      };
      prettier = {
        command = lib.getExe prettier;
        options = [ "--write" ];
        includes = [ "*.md" ];
        excludes = [ "flake.lock" ];
      };
      shellcheck = {
        command = lib.getExe shellcheck;
        includes = [ "*.sh" ];
        excludes = [ ".envrc" ];
      };
      shfmt = {
        command = lib.getExe shfmt;
        options = [
          "-w"
          "-i"
          "2"
          "-s"
        ];
        includes = [
          "*.sh"
          "*.envrc"
        ];
      };
    };
  };
}
