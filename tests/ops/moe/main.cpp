#include "ops/suite.h"

namespace ninfer::test::moe {
int test_weights();
}

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    suite.add("moe/weight_contract", ninfer::test::moe::test_weights);
    return suite.main(argc, argv);
}
