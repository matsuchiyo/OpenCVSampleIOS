# OpenCVSampleIOS

reference:
- https://buildersbox.corp-sansan.com/entry/2023/02/13/110000
- https://github.com/opencv/opencv/blob/4.5.1/platforms/apple/readme.md

## Install cmake

```
% brew install cmake
```

## Install python

```
% brew install pyenv
% echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
% echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
% echo 'eval "$(pyenv init -)"' >> ~/.zshrc
% source ~/.zshrc
% pyenv install --list
% pyenv install 3.12.0
% pyenv global 3.12.0

```

## Build framework

```
% cd ~/work/ios
% git clone https://github.com/opencv/opencv.git
% python ~/work/ios/opencv/platforms/apple/build_xcframework.py \
  --iphoneos_archs arm64 \
  --iphonesimulator_archs x86_64,arm64 \
  --build_only_specified_archs \
  --iphoneos_deployment_target 13.0 \
  --out ~/work/ios/opencv_build_xcframework_py_output \
  --without video \
  --without dnn \
  --without gapi \
  --without stitching \
  --without ml \
  --without highgui \
  --without cudaimgproc \
  --without cudaarithm \
  --without cudalegacy \
  --without cudawarping \
  --without cudafeatures2d \
  --without photo \
  --without objdetect \
  --without ts \
  --without videoio
```

Copy ~work/ios/opencv_build_xcframework_py_output/opencv2.xcframework to ~/work/ios/OpenCVSampleIOS/OpenCVSampleIOS/frameworks.
Then, in [OpenCVSampleIOS > TARGETS OpenCVSampleIOS > General > Frameworks, Libraries and Embedded Content], add libc++.tbd and opencv2.framework (with "Do Not Embedded").
