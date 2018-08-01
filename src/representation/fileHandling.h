/*
 *	Utility code for file handling of various functions
        Implemented originally by Hui Lin, modified by Yuzong Liu and Kai Wei. Made more general by Rishabh Iyer.
 *
 */
#include "SparseGraph.h"
#include "SparseFeature.h"
#include "DenseFeature.h"
#include "Set.h"

namespace jensen {


long GetFileSize(const char * filename );

std::vector<double> readCostFunction(char* costFile, int n);

std::vector<double> readCostFunctionBinary(char* costFile, int n);

// A specific implementation to read a binary matrix stored as floats.
std::vector<std::vector<float> > readKernelfromFileFloat(char* graphFile, int n);

// A specific implementation to read a binary matrix stored as doubles.
std::vector<std::vector<float> > readKernelfromFileDouble(char* graphFile, int n);

// Helper Function to read in the features file and sparse graphs.
int line2wordsSparseGraph(char *s, struct SparseGraphItem & graphItem);
int line2words(char *s, struct SparseFeature & Feature);

// A specific implementation to read in a feature file, with number of features in first line.
std::vector<struct SparseFeature> readFeatureVectorsSparse(char* featureFile, int n, int &numFeatures);

std::vector<struct DenseFeature> readFeatureVectorsDense(string featureVectorPath, int n);

std::vector<struct SparseGraphItem> readSparseGraphfunction(char* sparseGraphFile, int n);

std::vector<HashSet> LoadTranscription(char* transcriptFile);

void LoadModularScore(char* scoreFile, std::vector<double>& weight, int& numNeighbors);

void writeResults(const char* strOutput, Set& set);

int checkline(char *line,unsigned num);

void printset(Set& set);
}
