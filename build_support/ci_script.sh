set -e
set -x
run_path=$(pwd)
echo "Cpplint code style check through shell"
# The run path should be the root of project
echo "run path: $run_path"

# default llvm-18 toolchain as default, please install it before use

src_path="tests"
include_path="include"

# check if src directory exists header file
for file in `find $src_path -name "*.h" -o -name "*.hpp"`; do
    # echo $file
    echo "Do not add header file in src directory"
    exit 1
done

# check if include directory exists cpp file
for file in `find $include_path -name "*.cpp"`; do
    echo "Do not add cpp file in include directory"
    exit 1
done

# do check for all cpp file in src directory
for file in `find $src_path -name "*.cpp"`; do
    python build_support/cpplint.py --filter=-legal/copyright --verbose=3  $file
    echo "Cpplint check $file finish"
done

# do check for all header file in include directory 
for file in `find $include_path  -name "*.h" -o -name "*.hpp"`; do
    python build_support/cpplint.py --filter=-legal/copyright --verbose=3  $file
    echo "Cpplint check $file finish"
done

echo "Cpplint check finish"

# clang format check
echo "Start clang-format check"
python build_support/run_clang_format.py --clang_format_binary clang-format-18 --exclude_globs ./build_support/ignore_files.txt --source_dir $src_path,$include_path
echo "Clang-format check finish"

# clang tidy check
echo "Start clang-tidy check"
python3 build_support/run_clang_tidy.py -clang-tidy-binary clang-tidy-18 -style file -p build
echo "clang-tidy check end"
