#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <xtorch/xtorch.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace fs = std::filesystem;
using namespace std;

int main(int argc, char* argv[])
{
    xt::datasets::CelebA celeb = xt::datasets::CelebA("/home/kami/Documents/datasets/");
    cout << celeb.size().value() << endl;
    cout << celeb.get(0).data.sizes() << endl;

    return 0;
}
