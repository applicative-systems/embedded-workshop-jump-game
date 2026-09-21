{
  description = "Embedded webcam post-controlled jump game demo on a Jetson Orin Nano appliance";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    jetpack-nixos = {
      url = "github:anduril/jetpack-nixos";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      eachSystem =
        systems: f:
        builtins.foldl' (
          a: s: a // builtins.mapAttrs (k: v: (a.${k} or { }) // { ${s} = v; }) (f s)
        ) { } systems;
    in
    {
      overlays.default = import ./overlay.nix;

      nixosModules = {
        # 1. Jetpack compatibility
        jetson = {
          imports = [
            inputs.jetpack-nixos.nixosModules.default
            ./modules/jetson.nix
          ];
          nixpkgs.overlays = [ inputs.jetpack-nixos.overlays.default ];
        };

        # 1. basic SSH, user, etc.
        base = ./modules/base.nix;

        # 2. + game on the screen. Our overlay rides along here rather than in
        #    jetson: everything in it (pose-weights, jump-game, the opencv4 and
        #    torch overrides) exists only to build this app.
        app = {
          imports = [ ./modules/app.nix ];
          nixpkgs.overlays = [ inputs.self.overlays.default ];
        };
        size-reduction = ./modules/size-reduction.nix; # 3. + drop what is unused
        boot-speed = ./modules/boot-speed.nix; # 4. + 39s -> 20s

        # 5. the disk topic: same software, two media.
        usb = ./hosts/usb.nix;
        nvme = {
          imports = [
            inputs.disko.nixosModules.disko
            ./hosts/nvme.nix
          ];
        };
      };

      nixosConfigurations =
        let
          m = inputs.self.nixosModules;
        in
        {
          nano-base = inputs.nixpkgs.lib.nixosSystem {
            modules = [
              m.jetson
              m.base
              m.usb
            ];
          };

          nano-app = inputs.nixpkgs.lib.nixosSystem {
            modules = [
              m.jetson
              m.base
              m.app
              m.usb
            ];
          };

          nano-slim = inputs.nixpkgs.lib.nixosSystem {
            modules = [
              m.jetson
              m.base
              m.app
              m.size-reduction
              m.usb
            ];
          };

          nano-fast = inputs.nixpkgs.lib.nixosSystem {
            modules = [
              m.jetson
              m.base
              m.app
              m.size-reduction
              m.boot-speed
              m.usb
            ];
          };

          # Same layers as nano-fast but on NVMe disk
          nano = inputs.nixpkgs.lib.nixosSystem {
            modules = [
              m.jetson
              m.base
              m.app
              m.size-reduction
              m.boot-speed
              m.nvme
            ];
          };
        };

    }
    // eachSystem systems (
      system:
      let
        inherit (inputs.nixpkgs) lib;
        pythonLibs = ps: [
          ps.ultralytics
          ps.opencv4
          ps.pytest
        ];
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          }
          // lib.optionalAttrs (system == "aarch64-linux") {
            cudaCapabilities = [ "8.7" ]; # nvidia jetson orin nano
          };
          overlays = [
            inputs.self.overlays.default
          ]
          ++ lib.optionals (system == "aarch64-linux") [
            inputs.jetpack-nixos.overlays.default
            (final: _: { cudaPackages = final.cudaPackages_13_2; })
          ];
        };

        configs = inputs.self.nixosConfigurations;
      in
      {
        packages = {
          inherit (pkgs) pose-weights;
        }
        // lib.optionalAttrs (system == "aarch64-linux") {
          default = pkgs.jump-game;
          inherit (pkgs) jump-game;

          image-base = configs.nano-base.config.system.build.image;
          image-app = configs.nano-app.config.system.build.image;
          image-slim = configs.nano-slim.config.system.build.image;
          image-fast = configs.nano-fast.config.system.build.image;

          usbImage = configs.nano-usb.config.system.build.image;
        };

        formatter = pkgs.callPackage ./treefmt.nix { };

        devShells.default = pkgs.mkShell {
          packages = [
            inputs.self.formatter.${system}
            (pkgs.python3.withPackages pythonLibs)
          ];
          shellHook = ''
            export POSE_WEIGHTS_DIR=${pkgs.pose-weights}
            export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
          '';
        };

        checks = {
          formatting = inputs.self.formatter.${system}.check inputs.self;
          # Declared with the package (jump-game/tests.nix), not here.
          inherit (pkgs.jump-game.tests) unit;
        };
      }
    );

  nixConfig = {
    # most of the CUDA packages aren't cached because we need them with
    # jetson capabilities which are not in default nixpkgs
    extra-substituters = [ "https://cache.nixos-cuda.org" ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };
}
