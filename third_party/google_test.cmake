FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2 # Use the desired release tag or commit hash
)
# Make the content available
FetchContent_MakeAvailable(googletest)
