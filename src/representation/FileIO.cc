/*
 *	Utility code for file handling of various functions
   Implemented originally by Hui Lin, modified by Yuzong Liu and Kai Wei. Made more general by Rishabh Iyer.
 *
 */
#include "FileIO.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "assert.h"
#include "../utils/error.h"

using namespace std;

namespace jensen {
#define MAXLEN 248

long GetFileSize(const char * filename )
{
	struct stat statebuf;

	if ( stat( filename, &statebuf ) == -1 )
		return -1L;
	else
		return statebuf.st_size;
}

Vector readVector(char* File, int n){
	Vector v;
	double tmpd;
	ifstream iFile;
	printf("Reading list of labels from %s...\n", File);
	iFile.open(File, ios::in);
	if (!iFile.is_open()) {
		printf("Error: Cannot open file\n");
	}
	else {
		for (int i=0; i<n; i++) {
			iFile >> tmpd;
			v.push_back(tmpd);
		}
	}
	iFile.close();
	return v;
}

Vector readVectorBinary(char* File, int n){
	Vector v;
	int unitsize = 8;
	double tmpd;
	FILE* fp;
	if(!(fp=fopen(File,"rb"))) {
		printf("ERROR: cannot open file %s",File);
	}

	for(int i=0; i<n; i++) {
		fread(&tmpd,unitsize,1,fp);
		v.push_back(tmpd);
	}
	fclose(fp);
	return v;
}

// A specific implementation to read a binary matrix stored as floats.
vector<vector<float> > readKernelfromFileFloat(char* graphFile, int n){
	vector<vector<float> > kernel;
	kernel.resize(n);
	for(int i=0; i<n; i++) {kernel[i].resize(n);}
	int unitsize=4;
	string tmp;
	ifstream iFile;
	int count = 0; // row count
	FILE* fp;
	float tmpf;
	printf("Loading graph from %s...\n",graphFile);
	if (!(fp=fopen(graphFile,"rb"))) {
		error("ERROR: cannot open file %s\n",graphFile);

	}
	int nRow = long(GetFileSize(graphFile)/n)/unitsize;
	printf("Number of rows: %d\n", nRow);
	for (int i = 0; i < nRow; i++) {
		for (int j = 0; j < n; j++) {
			fread(&tmpf,unitsize,1,fp);
			kernel[count+i][j] = tmpf;
		}
	}
	count += nRow;
	fclose(fp);
	printf("Finished loading the graph from %s...\n",graphFile);
	return kernel;
}

// A specific implementation to read a binary matrix stored as doubles.
vector<vector<float> > readKernelfromFileDouble(char* graphFile, int n){
	vector<vector<float> > kernel;
	kernel.resize(n);
	for(int i=0; i<n; i++) {kernel[i].resize(n);}
	int unitsize=8;
	string tmp;
	ifstream iFile;
	int count = 0; // row count
	FILE* fp;
	double tmpd;
	printf("Loading graph from %s...\n",graphFile);
	if (!(fp=fopen(graphFile,"rb"))) {
		printf("ERROR: cannot open file %s",graphFile);
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fread(&tmpd,unitsize,1,fp);
			kernel[i][j] = (float)tmpd;
		}
	}
	fclose(fp);
	return kernel;
}


// Helper Function to read in the Feature based functions.
int line2words(char *s, struct SparseFeature & Feature, int & max_feat_idx){
	int digitwrd;
	float featureval;
	int pos = 0;
	int numUniqueWords = 0;
	max_feat_idx = 0;
	while (sscanf(s,"%d %f %n",&digitwrd, &featureval, &pos) == 2) {
		s += pos;
		numUniqueWords++;
		if (digitwrd < 0) {
			cout << "The input feature graph is not right: " << " some feature index is <0, please make sure the feature indices being above 0\n";
		}
		assert((digitwrd >= 0));
		if (digitwrd > max_feat_idx) {
			max_feat_idx = digitwrd;
		}
		/*if (featureval < 0){
		   cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
		   }*/
		//assert(featureval >= 0);
		Feature.featureIndex.push_back(digitwrd);
		Feature.featureVec.push_back(featureval);
	}
	return numUniqueWords;
}

// A specific implementation to read in a feature file, with number of features in first line.
std::vector<struct SparseFeature> readFeatureVectorSparse(char* featureFile, int& n, int &numFeatures){
	// read the feature based function
	std::vector<struct SparseFeature> feats;
	// feats.resize(n);
	ifstream iFile;
	FILE *fp = NULL;
	char line[300000]; //stores information in each line of input
	printf("Reading feature File from %s...\n", featureFile);
	if ((fp = fopen(featureFile, "rt")) == NULL) {
		printf("Error: Cannot open file %s", featureFile);
		exit(-1);
	}
	long int lineno = 0;
	// read in the first line of the file, and skip it.
	//fgets(line,sizeof(line),fp);
	int pos = 0;
	n = 0;
	numFeatures = 0;
	//sscanf(line,"%d %d %n",&n,&numFeatures, &pos);
	//cout<<"n = "<<n<<" and "<< "numFeatures = "<<numFeatures<<"\n";
	int max_feat_idx;
	while ( fgets(line,sizeof(line),fp) != NULL) {
		feats.push_back(SparseFeature());
		feats[lineno].index = lineno;
		feats[lineno].numUniqueFeatures = line2words(line, feats[lineno], max_feat_idx); //line2words transforms input with fmt digwords:featurevals into initialization of structure "Feature"
		//feats[lineno].numFeatures = numFeatures;
		if (max_feat_idx > numFeatures) {
			numFeatures = max_feat_idx;
		}
		lineno++;
	}
	fclose(fp);
	n = lineno;
	numFeatures++;
	printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
	for (int idx = 0; idx < n; idx++) {
		feats[idx].numFeatures = numFeatures;
	}

	cout<<"done with reading the feature based file\n";
	return feats;
}

// Read labels and features stored in LIBSVM format.
void readFeatureLabelsLibSVM( const char* fname, std::vector<struct SparseFeature>& features, Vector& y, int& n, int &numFeatures)
{
	features = std::vector<struct SparseFeature>();
	FILE* file;
	printf("Reading feature File from %s...\n", fname);
	if ((file = fopen(fname, "rt")) == NULL) {
		printf("Error: Cannot open file %s", fname);
		exit(-1);
	}
	float label; bool init = true;
	char tmp[ 1024 ];
	numFeatures = 0;
	n = 0;
	struct SparseFeature feature;
	while( fscanf( file, "%s", tmp ) == 1 ) {
		int index; float value;
		if( sscanf( tmp, "%d:%f", &index, &value ) == 2 ) {
			feature.featureIndex.push_back(index); feature.featureVec.push_back(value);
			if (index > numFeatures)
				numFeatures = index;
		}else{
			if( !init ) {
				y.push_back(label);
				features.push_back(feature);
				feature = SparseFeature();
				n++;
			}
			assert(sscanf( tmp, "%f", &label ) == 1);
			init = false;
		}
	}
	y.push_back(label);
	features.push_back(feature);
	n++;
	numFeatures++;
	printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
	for (int idx = 0; idx < features.size(); idx++) {
		features[idx].numFeatures = numFeatures;
	}
	fclose(file);
}


// 2016-12-28
// A specific implementation to read in a feature file, with number of features in first line.
// Split dataset into a test set and training set
void readFeatureVectorSparseCrossValidate(char* featureFile, char* labelFile,
                                          int& numTrainingInstances, int &numFeatures,
                                          float percentTrain,
                                          std::vector<struct SparseFeature> &trainFeats,
                                          std::vector<struct SparseFeature> &testFeats,
                                          Vector &trainLabels,
                                          Vector &testLabels)
{
	// read the feature based function
	std::vector<struct SparseFeature> feats;
	// feats.resize(n);
	ifstream iFile;
	FILE *fp = NULL;
	char line[300000]; //stores information in each line of input
	printf("Reading feature File from %s...\n", featureFile);
	if ((fp = fopen(featureFile, "rt")) == NULL) {
		printf("Error: Cannot open file %s", featureFile);
		exit(-1);
	}
	long int lineno = 0;
	// read in the first line of the file, and skip it.
	//fgets(line,sizeof(line),fp);
	int pos = 0;
	int n = 0;
	numFeatures = 0;
	//sscanf(line,"%d %d %n",&n,&numFeatures, &pos);
	//cout<<"n = "<<n<<" and "<< "numFeatures = "<<numFeatures<<"\n";
	int max_feat_idx;
	while ( fgets(line,sizeof(line),fp) != NULL) {
		feats.push_back(SparseFeature());
		feats[lineno].index = lineno;
		feats[lineno].numUniqueFeatures = line2words(line, feats[lineno], max_feat_idx); //line2words transforms input with fmt digwords:featurevals into initialization of structure "Feature"
		//feats[lineno].numFeatures = numFeatures;
		if (max_feat_idx > numFeatures) {
			numFeatures = max_feat_idx;
		}
		lineno++;
	}
	fclose(fp);
	n = lineno;
	numFeatures++;
	printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
	for (int idx = 0; idx < n; idx++) {
		feats[idx].numFeatures = numFeatures;
	}

	cout<<"Done with reading the feature based file.\n";

	// Split into testing and training data
	// First gather labels
	Vector labels = readVector(labelFile, n);

	// if the number of instances were listed before the instances, we could do this in one pass
	numTrainingInstances = percentTrain * (float) n;

	cout <<"Splitting into " << numTrainingInstances << " training instances and " <<
	        n - numTrainingInstances << " testing instances.\n";

	std::vector<int> instances;
	for(int i = 0; i < n; i++) {
		instances.push_back(i);
	}
	srand(time(NULL));
	std::random_shuffle (instances.begin(), instances.end());

	// set aside first numTrainingInstances instances as training data, remaining n-numTrainingInstances instances
	// as test set
	int curr_feature_idx = 0;
	for (std::vector<int>::iterator it = instances.begin(); it != instances.end(); it++) {
		if( curr_feature_idx < numTrainingInstances) {
			trainFeats.push_back(feats[*it]);
			trainLabels.push_back(labels[*it]);
		} else {
			testFeats.push_back(feats[*it]);
			testLabels.push_back(labels[*it]);
		}
		curr_feature_idx++;
	}
}

}
