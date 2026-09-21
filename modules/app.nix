{ lib, pkgs, ... }:
let
  engineDir = "/var/lib/jump-game";
  autostart = pkgs.writeShellScript "jump-game-autostart" ''
    export POSE_ENGINE_DIR=${engineDir}
    while :; do
      ${pkgs.systemd}/bin/systemd-cat -t jump-game ${lib.getExe pkgs.jump-game}
      echo "jump-game exited $? -- restarting in 2s" \
        | ${pkgs.systemd}/bin/systemd-cat -t jump-game -p warning
      sleep 2
    done
  '';
in
{
  nixpkgs.config = {
    allowUnfree = true;
    cudaSupport = true;
    cudaCapabilities = [ "8.7" ]; # Jetson is not in nixpkgs defaults
  };

  services.xserver = {
    enable = true;
    windowManager.session = [
      {
        name = "jump-game";
        start = ''
          ${autostart} &
          waitPID=$!
        '';
      }
    ];
    displayManager.lightdm.enable = true;

    # Current demo HW setup requires us to pin this, otherwise
    # the 4K res only has 30Hz. Display + cable can actually do more.
    displayManager.setupCommands = ''
      ${pkgs.xrandr}/bin/xrandr --output DP-0 --mode 2560x1440 --rate 59.95
    '';

    serverFlagsSection = ''
      Option "BlankTime" "0"
      Option "StandbyTime" "0"
      Option "SuspendTime" "0"
      Option "OffTime" "0"
    '';

    moduleSection = ''
      Disable "dri"
    '';
  };
  services.displayManager = {
    defaultSession = "none+jump-game";
    autoLogin = {
      enable = true;
      user = "demo";
    };
  };

  users.users.demo = {
    isNormalUser = true;
    description = "Kiosk session running the jump game";
    initialPassword = "demo";
    extraGroups = [
      "video"
      "jump-game"
    ];
  };

  users.groups.jump-game = { };
  systemd.tmpfiles.rules = [ "d ${engineDir} 2775 demo jump-game -" ];

  environment.systemPackages = [ pkgs.jump-game ];

  environment.variables = {
    POSE_WEIGHTS_DIR = "${pkgs.pose-weights}";
    POSE_ENGINE_DIR = engineDir;
  };
}
