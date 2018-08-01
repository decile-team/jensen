/*
 *	Utility code for file handling of various functions
        Implemented originally by Hui Lin, modified by Yuzong Liu and Kai Wei. Made more general by Rishabh Iyer.
 *
 */
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "error.h"
#include "assert.h"

using namespace std;

#include "fileHandling.h"

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

vector<double> readCostFunction(char* costFile, int n){
	vector<double> costList;
	double tmpd;
	ifstream iFile;
	printf("Reading list of costs from %s...\n", costFile);
	iFile.open(costFile, ios::in);
	if (!iFile.is_open()) {
		printf("Use unit cost for each item\n");
		for (int i=0; i<n; i++) {
			costList.push_back(1.0);
		}
	}
	else {
		for (int i=0; i<n; i++) {
			iFile >> tmpd;
			costList.push_back(tmpd);
		}
	}
	iFile.close();
	return costList;
}

vector<double> readCostFunctionBinary(char* costFile, int n){
	vector<double> costList;
	int unitsize = 8;
	double tmpd;
	FILE* fp;
	if(!(fp=fopen(costFile,"rb"))) {
		printf("ERROR: cannot open file %s",costFile);
	}

	for(int i=0; i<n; i++) {
		fread(&tmpd,unitsize,1,fp);
		costList.push_back(tmpd);
	}
	fclose(fp);
	return costList;
}

void writeResults(const char* strOutput, Set& set){
	FILE* fp = NULL;
	if (!(fp=fopen(strOutput, "wt"))) {
		error("Cannot open the file %s for output\n", strOutput);
	}
	Set::iterator it(set);
	for (it = set.begin(); it != set.end(); ++it) {
		fprintf(fp, "%d ", *it);
	}
	fclose(fp);
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

// Helper Function to read in the sparse graph.
int line2wordsSparseGraph(char *s, struct SparseGraphItem & graphItem, int n){
	int digitwrd;
	float featureval;
	int pos = 0;
	int num_words = 0;
	while (sscanf(s,"%d %f %n",&digitwrd, &featureval, &pos) == 2) {
		s += pos;
		num_words++;
		if ((digitwrd >= n) || (digitwrd < 0)) {
			cout << "The input sparse graph is not right: " << " the nearest neighbor index is >= n or <0, please make sure the sparse graph has indices ranging from 0 to n-1\n";
		}
		assert((digitwrd < n) && (digitwrd >= 0));
		if (featureval < 0) {
			cout << "The input sparse graph has a similarity value being negative, please make sure that the sparse graph has all similarity value >= 0\n";
		}
		assert(featureval>=0);
		graphItem.NNIndex.push_back(digitwrd);
		graphItem.NNSim.push_back(featureval);
		//cout<<digitwrd<<" "<<featureval<<" ";
	}
	//cout<<"\n";
	return num_words;
}
// Helper Function to read in the Feature based functions.
int line2words(char *s, struct SparseFeature & Feature, int nFeatures){
	int digitwrd;
	float featureval;
	int pos = 0;
	int num_words = 0;
	while (sscanf(s,"%d %f %n",&digitwrd, &featureval, &pos) == 2) {
		s += pos;
		num_words++;
		if ((digitwrd >= nFeatures) || (digitwrd < 0)) {
			cout << "The input feature graph is not right: " << " some feature index is either >= nFeatures or <0, please make sure the feature indices ranging from 0 to nFeatures-1\n";
		}
		assert((digitwrd < nFeatures) && (digitwrd >= 0));
		if (featureval < 0) {
			cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
		}
		assert(featureval >= 0);
		Feature.featureIndex.push_back(digitwrd);
		Feature.featureVec.push_back(featureval);
	}
	return num_words;
}

// A specific implementation to read in a sparse similarity graph, with the ground set size specified in first line.
vector<struct SparseGraphItem> readSparseGraphfunction(char* featureFile, int n){
// read the feature based function
	vector<struct SparseGraphItem> feats;
	//feats.resize(n);
	ifstream iFile;
	FILE *fp = NULL;
	char line[300000]; //stores information in each line of input
	// need to add checkline whether a line has more than 300000 characters.
	printf("Reading feature File from %s...\n", featureFile);
	if ((fp = fopen(featureFile, "rt")) == NULL) {
		printf("Error: Cannot open file %s", featureFile);
		exit(-1);
	}
	long int lineno = 0;
	// read in the first line of the file, and skip it.
	fgets(line,sizeof(line),fp);
	int pos = 0;
	sscanf(line,"%d %n",&n, &pos);
	cout<<"n = "<<n<<"\n";
	while ( fgets(line,sizeof(line),fp) != NULL) {
		feats.push_back(SparseGraphItem());
		feats[lineno].index = lineno;
		feats[lineno].num_NN_items = line2wordsSparseGraph(line, feats[lineno], n); //line2wordsSparseGraph transforms input with fmt digwords:featurevals into initialization of structure "sparse graph"
		lineno++;
	}
	fclose(fp);
	if (n != lineno) {
		printf("error: number of points doesn't match with input number of points\n");
		exit(-1);
	}
	cout<<"done with reading the sparse graph file\n";
	return feats;
}

// A specific implementation to read in a feature file, with number of features in first line.
vector<struct SparseFeature> readFeatureVectorsSparse(char* featureFile, int n, int &numFeatures){
// read the feature based function
	vector<struct SparseFeature> feats;
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
	fgets(line,sizeof(line),fp);
	int pos = 0;
	sscanf(line,"%d %d %n",&n,&numFeatures, &pos);
	cout<<"n = "<<n<<" and "<< "numFeatures = "<<numFeatures<<"\n";
	while ( fgets(line,sizeof(line),fp) != NULL) {
		feats.push_back(SparseFeature());
		feats[lineno].index = lineno;
		feats[lineno].num_uniq_wrds = line2words(line, feats[lineno], numFeatures); //line2words transforms input with fmt digwords:featurevals into initialization of structure "Feature"
		lineno++;
	}
	fclose(fp);
	if (n != lineno) {
		printf("error: number of points doesn't match with input number of points\n");
		exit(-1);
	}
	cout<<"done with reading the feature based file\n";
	return feats;
}

vector<struct DenseFeature> readFeatureVectorsDense(string featureVectorPath, int n) {
	int nrImages = n;
	int maxLength = 10000; // maximum length of feature vectors
	std::vector<struct DenseFeature> featureVectors;

	int unitsize = sizeof(float);

	int featureVectorLength = 0;
	for(int image = 0; image < nrImages; image++) {
		ifstream iFile;
		FILE* fp;
		string filename = featureVectorPath + "img" + to_string((long long)image+1) + ".vec";
		if (!(fp=fopen(filename.c_str(),"rb"))) {
			throw runtime_error("ERROR: cannot open file.\n");
		}

		float cache[2*maxLength];
		int i = 0;
		int length = fread(cache, unitsize, maxLength, fp);
		if(length == maxLength) {
			throw runtime_error("ERROR: Feature vector too long.\n");
		}
		if(image == 0) {
			featureVectorLength = length;
		} else {
			// santiy check: all feature vectors should have the same length
			if(featureVectorLength != length) {
				throw runtime_error("ERROR: Feature vectors do not have the same length.\n");
			}
		}

		std::vector<double> tvector(length);
		for (int j = 0; j < length; j++) {
			tvector[j] = (double) cache[j];
		}
		struct DenseFeature tvectorstruct;
		tvectorstruct.index = image;
		tvectorstruct.featureVec = tvector;
		featureVectors.push_back(tvectorstruct);

		fclose(fp);
	}

	return featureVectors;
}

std::vector<HashSet> LoadTranscription(char* transcriptFile)
{
	cout << "Start reading the transcription file" << endl;
	std::vector<HashSet> neighbors;
	ifstream iFile(transcriptFile);
	string line;
	neighbors.clear();
	if (iFile.is_open())
	{
		HashSet trans;
		int index = 0;
		while (getline(iFile, line))
		{
			istringstream currentline(line);
			trans.clear();
			string uttr_id;
			int token;
			currentline >> uttr_id;
			while (currentline >> token)
			{
				trans.insert(token);
			}
			neighbors.push_back(trans);
			index++;
			//cout << line << endl;
			//cout << index << endl;
		}
	}
	cout << "Done reading the transcription file" << endl;
	return neighbors;
}

void LoadModularScore(char* scoreFile, std::vector<double>& weights, int& numNeighbors)
{

	ifstream iFile(scoreFile);
	string line;
	weights.clear();
	if (iFile.is_open())
	{
		while (getline(iFile,line))
		{
			istringstream currentline(line);
			double currweight;
			currentline >> currweight;
			weights.push_back(currweight);
		}
	}

	numNeighbors = weights.size();
}

int checkline(char *line,unsigned num) {

	if (strlen(line) == MAXLEN && line[strlen(line)-1] != '\n') {
		error("Line %li too long\n",num);
	}
	else {
		line[strlen(line)-1] = '\0';
	}
	return strlen(line);
}

void printset(Set& sset){
	Set::iterator it(sset);
	for( it = sset.begin(); it != sset.end(); ++it ) {
		cout<<*it<<" ";
	}
	cout<<"\n";
}


}
