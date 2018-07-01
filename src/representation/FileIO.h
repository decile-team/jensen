/*
 *	Utility code for file handling of various functions
 Author: Rishabh Iyer.
 *
 */

#ifndef Jensen_FILE_IO
#define Jensen_FILE_IO

#include "Vector.h"
#include "SparseFeature.h"


namespace jensen {

  long GetFileSize(const char * filename);

  Vector readVector(char* File, int n);

  Vector readVectorBinary(char* File, int n);

  // A specific implementation to read a binary matrix stored as floats.
  std::vector<std::vector<float> > readKernelfromFileFloat(char* graphFile, int n);

  // A specific implementation to read a binary matrix stored as doubles.
  std::vector<std::vector<float> > readKernelfromFileDouble(char* graphFile, int n);

  int line2words(char *s, struct SparseFeature & Feature, int& maxID);

  std::vector<struct SparseFeature> readFeatureVectorSparse(char* featureFile, int& n, int &numFeatures);

  void readFeatureLabelsLibSVM( const char* fname, std::vector<struct SparseFeature>& features, Vector& y, int& n, int &numFeatures);

  void readFeatureVectorSparseCrossValidate(char* featureFile, char* labelFile,
					    int& numTrainingInstances, int& numFeatures, 
					    float percentTrain,
					    std::vector<struct SparseFeature>& trainFeats, 
					    std::vector<struct SparseFeature>& testFeats,
					    Vector& trainLabels, 
					    Vector& testLabels);

}
#endif
