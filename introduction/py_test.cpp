#include <iostream>
#include <string>
#include <filesystem> // Requires C++17

// Platform-specific includes to find the library's path at runtime
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

/**
 * @brief A struct to hold all the important paths for the xTorch library.
 */
struct XTorchPaths {
    fs::path install_prefix;      // The root installation directory (e.g., /usr/local)
    fs::path library_path;        // Full path to libxTorch.so
    fs::path share_dir;           // Path to the 'share/xtorch' directory
    fs::path venv_dir;            // Path to the 'venv' directory
    fs::path python_executable;   // Path to the python interpreter inside the venv
    fs::path python_modules_dir;  // Path to your custom python scripts
    bool found = false;           // A flag to indicate if paths were successfully located
};

/**
 * @brief Finds the path of the currently running shared library (libxTorch.so).
 *
 * This function is the key to making the entire library relocatable.
 * @return The full path to this library, or an empty string on failure.
 */
std::string get_this_library_path() {
#ifdef _WIN32
    // Windows implementation
    char path[MAX_PATH];
    HMODULE hm = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&get_this_library_path, &hm) == 0) {
        return ""; // Failed
    }
    if (GetModuleFileNameA(hm, path, sizeof(path)) == 0) {
        return ""; // Failed
    }
    return std::string(path);
#else
    // Linux, macOS, and other POSIX-like systems implementation
    Dl_info info;
    if (dladdr((void*)&get_this_library_path, &info) != 0) {
        return std::string(info.dli_fname);
    }
    return ""; // Failed
#endif
}

/**
 * @brief Locates all essential xTorch paths relative to the library's location.
 *
 * This should be the first utility function you call before using any Python
 * features of the library.
 *
 * @return An XTorchPaths struct populated with the discovered paths.
 */
XTorchPaths find_library_paths() {
    XTorchPaths paths;

    // 1. Get the path to our own library file (e.g., /usr/local/lib/libxTorch.so)
    fs::path lib_path = get_this_library_path();
    if (lib_path.empty()) {
        std::cerr << "xTorch ERROR: Could not determine its own library path." << std::endl;
        return paths; // Return with paths.found = false
    }
    paths.library_path = lib_path;

    // 2. Calculate all other paths relative to the library's path
    //    <prefix>/lib/libxTorch.so -> <prefix>
    paths.install_prefix = lib_path.parent_path().parent_path();

    //    <prefix> -> <prefix>/share/xtorch
    paths.share_dir = paths.install_prefix / "share" / "xtorch";

    //    .../share/xtorch -> .../share/xtorch/venv
    paths.venv_dir = paths.share_dir / "venv";

    //    .../venv -> .../venv/bin/python
    paths.python_executable = paths.venv_dir / "bin" / "python";

    //    .../share/xtorch -> .../share/xtorch/py_modules
    paths.python_modules_dir = paths.share_dir / "py_modules";

    // 3. Final check: verify the venv directory exists as a sanity check.
    if (!fs::exists(paths.venv_dir)) {
        std::cerr << "xTorch WARNING: Venv directory not found at expected location: "
                  << paths.venv_dir << std::endl;
        std::cerr << "xTorch WARNING: Please ensure 'make install' completed successfully." << std::endl;
        return paths; // Return with paths.found = false
    }

    paths.found = true;
    return paths;
}

// --- Example of how to use it ---
int main() {
    std::cout << "--- Finding xTorch Library Paths ---" << std::endl;

    XTorchPaths paths = find_library_paths();

    if (paths.found) {
        std::cout << "Successfully located library components." << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        auto check_exists = [](const fs::path& p) {
            return fs::exists(p) ? " [found]" : " [MISSING]";
        };

        std::cout << "Install Prefix:      " << paths.install_prefix << check_exists(paths.install_prefix) << std::endl;
        std::cout << "Library Path:        " << paths.library_path << check_exists(paths.library_path) << std::endl;
        std::cout << "Share Directory:     " << paths.share_dir << check_exists(paths.share_dir) << std::endl;
        std::cout << "Venv Directory:      " << paths.venv_dir << check_exists(paths.venv_dir) << std::endl;
        std::cout << "Python Executable:   " << paths.python_executable << check_exists(paths.python_executable) << std::endl;
        std::cout << "Python Modules Dir:  " << paths.python_modules_dir << check_exists(paths.python_modules_dir) << std::endl;

    } else {
        std::cerr << "Failed to locate the required library paths." << std::endl;
        return 1;
    }

    return 0;
}