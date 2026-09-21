{
  lib,
  runCommand,
  python,
}:

runCommand "jump-game-tests"
  {
    nativeBuildInputs = [
      (python.withPackages (ps: [
        ps.pytest
        ps.numpy
        ps.opencv4
      ]))
    ];

    src = lib.fileset.toSource {
      root = ./.;
      fileset = lib.fileset.unions [
        ./src
        ./tests
        ./pytest.ini
        ./taunts.json
        ./images
      ];
    };
  }
  ''
    cp -r $src/* .
    pytest -q
    touch $out
  ''
