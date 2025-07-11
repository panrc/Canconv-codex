pip install -r requirements.txt
git submodule update --init --recursive
mkdir -p build
cmake -DCMAKE_BUILD_TYPE:STRING=Release -S . -B build -G Ninja
cmake --build build --config Release --target all