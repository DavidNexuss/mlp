#pragma once
#include <net/net.h>
#include <net/activation.hpp>

std::vector<std::shared_ptr<MLPTrainer>> getTests();
std::shared_ptr<MLPTrainer>              getTests(int index);
