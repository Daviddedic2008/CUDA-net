#pragma once

extern "C" void launchKernel(const unsigned int numBlocks, const unsigned int numThreads); // ensure compatibility w/ cpp linker

extern "C" void testVector();
