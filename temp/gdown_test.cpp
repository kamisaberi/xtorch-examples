#include <xtorch/xtorch.h>
#include <iostream>

using namespace std;

int main()
{
    // xt::utils::download_from_google_drive("0B7EVK8r0v71pblRyaVFSWGxPY0U", "00000000000000000000000000000000" , "/home/kami/temp/list_attr_celeba.txt");
    auto dataset = xt::datasets::CelebA("/home/kami/temp" , xt::datasets::DataMode::TRAIN , true);
    return 0;
}
