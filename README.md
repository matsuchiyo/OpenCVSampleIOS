# OpenCVSampleIOS

reference: https://qiita.com/treastrain/items/0090d1103033b20de054

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
% python opencv/platforms/ios/build_framework.py opencv_build_framework_py_output \
  # ↓ error: The armv7 architecture is deprecated. You should update your ARCHS build setting to remove the armv7 architecture. (in target 'cmTC_d46dc' from project 'CMAKE_TRY_COMPILE')
  --iphoneos_archs arm64 \
  --iphonesimulator_archs x86_64
```

Copy ~work/ios/opencv_biuld_framek_py_output/opencv2.framework to ~/work/ios/OpenCVSampleIOS/OpenCVSampleIOS/frameworks.
Then, in [OpenCVSampleIOS > TARGETS OpenCVSampleIOS > General > Frameworks, Libraries and Embedded Content], add libc++.tbd and opencv2.framework (with "Do Not Embedded").
