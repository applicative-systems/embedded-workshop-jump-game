{
  lib,
  buildPythonApplication,
  setuptools,

  numpy,
  opencv4,
  pillow,
  torch,
  ultralytics,
  onnx,
  onnxslim,
  tensorrt,

  dejavu_fonts,
  ffmpeg,
  xrandr,
  pose-weights,

  callPackage,
}:

buildPythonApplication {
  pname = "jump-game";
  version = "1.0.0";
  pyproject = true;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./src
      ./images
      ./taunts.json
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    numpy
    opencv4
    pillow
    torch
    ultralytics
    onnx
    onnxslim
    tensorrt
  ];

  postInstall = ''
    mkdir -p $out/share/jump-game
    cp -r images taunts.json $out/share/jump-game/
  '';

  makeWrapperArgs = [
    "--set"
    "JUMP_ASSETS_DIR"
    "$out/share/jump-game/images"
    "--set"
    "JUMP_TAUNTS"
    "$out/share/jump-game/taunts.json"
    "--set"
    "POSE_WEIGHTS_DIR"
    "${pose-weights}"
    "--set"
    "JUMP_FONT"
    "${dejavu_fonts}/share/fonts/truetype/DejaVuSans-Bold.ttf"
    "--prefix"
    "PATH"
    ":"
    (lib.makeBinPath [
      ffmpeg
      xrandr
    ])
  ];

  pythonImportsCheck = [
    "coco"
    "duck"
    "game"
    "gameover"
    "jump_detector"
    "loop"
    "record"
    "sprite"
    "takeoff"
  ];

  passthru.tests.unit = callPackage ./tests.nix { };

  meta = {
    description = "Webcam pose-controlled jump-and-duck game";
    mainProgram = "jump-game";
  };
}
