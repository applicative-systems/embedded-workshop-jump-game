final: prev: {
  # we would not need *all* the weights for a slick deployment.
  # added all of them for quality comparison
  pose-weights = final.linkFarm "pose-weights" (
    map
      ({ name, hash }: {
        name = "${name}.pt";
        path = final.fetchurl {
          url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/${name}.pt";
          inherit hash;
        };
      })
      [
        {
          name = "yolo26n-pose";
          hash = "sha256-6zu4Jogorq9RXOwjpL+v15OUSob+mvlLp4I2CcFFIqk=";
        }
        {
          name = "yolo26s-pose";
          hash = "sha256-oIOttCMDcorhTEvWvVbYDaRvgvslZNvW8x3MkuoyFkY=";
        }
        {
          name = "yolo26m-pose";
          hash = "sha256-L78WNnAiJWoiYDVpXFw4k4TGcG6LuKuPzQ55dvBUQ8Q=";
        }
        {
          name = "yolo26l-pose";
          hash = "sha256-rTPaiinqV3IxjEyYCETke1Z5LStjgVrU6OCcB4x9Gr8=";
        }
        {
          name = "yolo26x-pose";
          hash = "sha256-CO2eAdIqbySLBPL5mSAWrKmjIlC5q1cFfYhqCdAmcA0=";
        }
      ]
  );

  opencv4 = prev.opencv4.override { enableGtk3 = true; };

  # No NCCL on Jetson. transitively pulled via torch.
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (_pyfinal: pyprev: {
      torch = pyprev.torch.override { withNvshmem = false; };
    })
  ];

  jump-game = final.python3Packages.callPackage ./jump-game { };
}
