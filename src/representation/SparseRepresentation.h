// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#include "../external/db.h"
#include <unordered_map>
#include <vector>
#include <string>
#include "SparseFeature.h"
#ifndef Jensen_FEATUREREP
#define Jensen_FEATUREREP

// inputFileType has to be either 1 (text), or 2 (levelDB+string), or 3 (levelDB+floatBinary), or 4 (levelDB+doubleBinary) and out of core has to be either 0 (in core) or 1 (out of core); and out of core can only be 1 if inputFileType is levelDB type (i.e., 2,3,4)")

namespace smtk {

	class SparseRepresentation{
	private:
		std::vector<struct SparseFeature> features; // A list of features stored in main memory
		int nFeatures; // Total Number of features.
		SparseFeature currfeature; // A current feature, used if the feature if out of core (in which case, we load the feature to this, and return this item)
		int curridxpoint; // stores the item which defines currfeature.
		int n; // the size of the feature representation, i.e the size of the groundset. This could be larger than feats.size(), if not all data is in the main memory.
		std::unordered_map<int, int> currSetMap; // A map to the current set, which maps every element in the current set to a unique no. in 0, .., currSet.size() - 1
		const char* featureFile; // path pointing to the location of the feature file on disk. This will be needed if the data needs to be reloaded.
		const int inputFileType; // whether the features are stored in a text file, a levelDB database etc. etc.
		// 1: TextFile, 2:levelDB + text string, 3:levelDB + float binary, 4:levelDB + double binary 
		const bool outOfCore; // 0 if everything is in memory, and 1 if it is out of core.
		// Database pointers
		const leveldb::Options options;
		leveldb::DB* db;
	public:
		int line2Features(std::string line, struct SparseFeature & Feature, int nFeatures, bool safe = 0); // A helper function to read in a line of a text or DB file!
		SparseRepresentation(const char* featureFile, int inputFileType, bool outOfCore); // a constructor 
		//SparseRepresentation(char* levelDBFile);
		~SparseRepresentation();
		void readFeaturesText(const char* featureFile, int verbosity = 1); // reads the feature representation from a text file
		void readFeaturesLevelDB(const char* levelDBFile, int verbosity = 1, int mode = 0); // reads the feature representation from a levelDB file, mode = 0, data is stored as string, mode = 1, data is stored as floats, mode = 2, data is stored as double.
		int size(); // returns n
		int numFeatures(); // returns nFeatures
		SparseFeature& getFeature(int item); // returns the feature corresponding to item
		SparseFeature& operator[](int item);
		void resetRepresentationSet(const std::vector<int>& set);
        void closeDB();
		//void resetRepresentationRandom(unordered_set<int>& csset, int rsize, unordered_set<int>& randomSet);
		//void resetRepresentationSequential(int startitem, int setsize, unordered_set<int>& outSet);
	};
    
    void writeFeaturesLevelDBBinary(char* outputlevelDBFile, char* inputlevelDBFile, int mode, int verbosity); // transform the level DB data which is in text format to level DB data in binary format.

}

#endif
