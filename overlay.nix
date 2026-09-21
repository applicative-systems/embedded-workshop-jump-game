final: prev: {
  # we would not need *all* the weights for a slick deployment.
  # added all of them for quality comparison
  pose-weights = final.linkFarm "pose-weights" (
    map
      ({ name, hash }: {
        name = "${name}.pt";
        path = final.fetchurl {
          url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/${name}.pt";
          inherit hash;
        };
      })
      [
        {
          name = "yolo11n-pose";
          hash = "sha256-hp6D/N/9xzcfpONM2OUcg4zHKVcdFjXlFB4wdekxncA=";
        }
        {
          name = "yolo11s-pose";
          hash = "sha256-EGC9pKJwEgYOyiRvmyre6iLquwRaHlj40im+Kbfrwro=";
        }
        {
          name = "yolo11m-pose";
          hash = "sha256-KbF+rzoxF8vqkGCQ2+35FZ98aknbWOyLme0t/eHPbrI=";
        }
        {
          name = "yolo11l-pose";
          hash = "sha256-YZIavh8u2TC/KDKKFrMWInjNsjmh6CgLasgnEZ8TzuA=";
        }
        {
          name = "yolo11x-pose";
          hash = "sha256-ATxDVDsHUbiRhIa6luAe5EpZBAaDoH+WzCK8wst3hfg=";
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
