// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <iomanip>
#include "assert.h"
using namespace std;

#include "../external/db.h"
#include "SparseRepresentation.h"
#include "../external/write_batch.h"
#include "misc.h"
#include "../utils/error.h"

namespace smtk {

	SparseRepresentation::SparseRepresentation(const char* featureFile, const int inputFileType, const bool outofCore): featureFile(featureFile), inputFileType(inputFileType), outOfCore(outofCore)
    {
        features.clear();
        if( (inputFileType == 1) && (outOfCore == 0) ){
            readFeaturesText(featureFile); // in core with text file as the feature file. 
        }
        else if ( (inputFileType == 1) && (outOfCore == 1) )
        {
            error("Error: outofCore not supported with text files\n");
        }
        else if((inputFileType == 2)&& (outOfCore == 0)){ // load everything inCore
            int verbosity = 1;
            readFeaturesLevelDB(featureFile, verbosity, 0); // input file is levelDB + text string
        }
        else if ( (inputFileType == 3) && (outOfCore == 0)){// load everything inCore  
            int verbosity = 1;
            readFeaturesLevelDB(featureFile, verbosity, 1);// input file is levelDB + float binary
        }
        else if ( (inputFileType == 4) && (outOfCore == 0)){ // load everything inCore
            int verbosity = 1;
            readFeaturesLevelDB(featureFile, verbosity, 2); // input file is levelDB + double binary 
        }
        else if(((inputFileType == 2) || (inputFileType == 3) || (inputFileType == 4)) && (outOfCore == 1)) // if the input file is levelDB with either text string, fload binary, or double binary, and outofCore is set on. 
        {
            leveldb::Status s = leveldb::DB::Open(options, featureFile, &db);
            if (!s.ok()){
                cerr<<"DB error:"<<s.ToString()<<"\n";
                exit(-1);
            }
            leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
            it->SeekToFirst(); 
            string key = it->key().ToString();
            string line = it->value().ToString();
            int pos = 0;
            sscanf(line.c_str(),"%d %d %n",&n,&nFeatures, &pos);
            cout<<"n = "<<n<<" and "<< "numFeatures = "<<nFeatures<<"\n";
            curridxpoint = n+1;
        }
        else
            fprintf(stderr, "Unknown inputFileType has to be either 1 (text), or 2 (levelDB+string), or 3 (levelDB+floatBinary), or 4 (levelDB+doubleBinary) and out of core has to be either 0 (in core) or 1 (out of core); and out of core can only be 1 if inputFileType is levelDB type (i.e., 2,3,4)");
    }

	SparseRepresentation::~SparseRepresentation(){
		if (outOfCore){
			delete db;
		}
	}
    void SparseRepresentation::closeDB(){
        if (outOfCore){
            delete db;
        }
    }
	int SparseRepresentation::size(){
		return n;
	}
	
	int SparseRepresentation::numFeatures(){
		return nFeatures;
	}

	void SparseRepresentation::resetRepresentationSet(const vector<int>& currSet)
    {
        if (outOfCore)
        {
            // Reset the features by loading in the elements of sset, from the DB.
            // Define a map corresponding to currSet and clear it.
            currSetMap.clear();
            int count = 0;
            // The code to read from a levelDB
            features.clear();
            features.resize(currSet.size());
            // leveldb::Options options;
            // leveldb::DB* db;
            // leveldb::Status s = leveldb::DB::Open(options, featureFile, &db);	
            leveldb::Status s;	
            for (vector<int>::const_iterator it= currSet.begin(); it!=currSet.end(); it++)
            {
                currSetMap[*it] = count;
                stringstream ss;
                ss << setw(10) << setfill('0') << *it+1;
                string key = ss.str();
                string value;
                s = db->Get(leveldb::ReadOptions(), key, &value);
                features[count].index = *it;
                if (inputFileType == 2){
                    features[count].num_uniq_wrds = line2Features(value, features[count], nFeatures);
                }
                else if (inputFileType == 3){
                    features[count].num_uniq_wrds = line2FeaturesFloat(value, features[count], nFeatures,true);
                }
                else if (inputFileType == 4){
                    features[count].num_uniq_wrds = line2FeaturesDouble(value, features[count], nFeatures, true);
                }
                count++;
            }
        }
        else
            error("Reset Representation should only be called when out of Core\n");
        // delete db;
    }

	// Helper Function to read in the Feature based functions.
	int SparseRepresentation::line2Features(string line, struct SparseFeature & Feature, int nFeatures, bool safe){
		int mode = 1;
		if (mode == 0)
			return string2Features(line, Feature, nFeatures, safe);
		else if(mode == 1)
			return char2Features(line.c_str(), Feature, nFeatures, safe);
		else if(mode == 2)
			return string2FeaturesFast(line.c_str(), Feature, nFeatures, safe);
		else{
			fprintf(stderr, "Mode not supported. Mode should be either 1, 2 or 3.\n");
            exit (-1);
		}
	}
    
	
	void SparseRepresentation::readFeaturesText(const char* featureFile, int verbosity)
	{
		// feats.resize(n);
		ifstream iFile(featureFile);
		string line; //stores information in each line of input
		if (verbosity > 0) printf("Reading feature File from %s...\n", featureFile);
		if (iFile.is_open()){
			long int lineno = 0;
			// read in the first line of the file, and skip it.
			getline(iFile,line);
			int pos = 0;
			sscanf(line.c_str(),"%d %d %n",&n,&nFeatures, &pos);
			cout<<"n = "<<n<<" and "<< "numFeatures = "<<nFeatures<<"\n";
			while (getline(iFile,line)){
				// getline(iFile,line);
			    	features.push_back(SparseFeature());     
			    	features[lineno].index = lineno;
			    	features[lineno].num_uniq_wrds = line2Features(line, features[lineno], nFeatures); //line2Features transforms input with fmt digwords:featurevals into initialization of structure "Feature"    
			    	lineno ++;
			}
			if (n != lineno){
			    fprintf(stderr, "error: the number of lines in the file doesn't match with input number of points\n");
			    exit(-1);
			}
			if (verbosity > 0) cout<<"done with reading the feature based file\n";
		}
		else{
		    fprintf(stderr, "Error: Cannot open file %s", featureFile);
		    exit(-1);
		}		
	}

	
	void SparseRepresentation::readFeaturesLevelDB(const char* levelDBFile, int verbosity, int mode){ // reads the feature representation from a levelDB file entirely.
		string line; //stores information in each line of input
		string key;
		if (verbosity > 0) printf("Reading feature File from %s...\n", featureFile);
		int lineno = 0;
		leveldb::Status s = leveldb::DB::Open(options, featureFile, &db);
		if (!s.ok()){
			cerr<<"DB error:"<<s.ToString()<<"\n";
            exit(-1);
		}
	  	leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
	 	 for (it->SeekToFirst(); it->Valid(); it->Next()) {
	    			key = it->key().ToString();
				line = it->value().ToString();
				if (lineno == 0){
					int pos = 0;
					sscanf(line.c_str(),"%d %d %n",&n,&nFeatures, &pos);
                    //cout << key << endl;
                    //cout << line << endl;
					cout<<"n = "<<n<<" and "<< "numFeatures = "<<nFeatures<<"\n";
					lineno++;
				}
				else{
			    		features.push_back(SparseFeature());     
			    		features[lineno-1].index = lineno;
                        if (mode == 0){
			    		    features[lineno-1].num_uniq_wrds = line2Features(line, features[lineno-1], nFeatures); //line2Features transforms input with fmt digwords:featurevals into initialization of structure "Feature"    
                        }
                        else if (mode == 1){
			    		    features[lineno-1].num_uniq_wrds = line2FeaturesFloat(line, features[lineno-1], nFeatures, true); //line2Features transforms input with fmt digwords:featurevals into initialization of structure "Feature"    
                        }
                        else if (mode == 2){
			    		    features[lineno-1].num_uniq_wrds = line2FeaturesDouble(line, features[lineno-1], nFeatures, true); //line2Features transforms input with fmt digwords:featurevals into initialization of structure "Feature"    
                        }
                        else{
                            cerr << "Incorrect mode for reading the levelDB file: only support 0: text, 1: float binary, and 2: double binary" << endl;
                            exit(-1);
                        }
			    		lineno ++;
				}	
	  	}	
		delete db;
	}
	
	SparseFeature& SparseRepresentation::getFeature(int item){ // make sure item < n (this is not checked in the code)
        //cout << "Try to access " << item << endl<<flush;
		if (outOfCore == 0) // everything in memory
			return features[item];
		else // things could be out of core.
		{
			if(currSetMap.find(item) != currSetMap.end()){ // item belongs to the current set
				return features[currSetMap[item]];
			}
			else{ // item does not belong to the current set, we will need to get it from the DB.
				//cout<<"Data is out of memory, accessing from DB\n"<<flush;
                if (curridxpoint == item){
					return currfeature;
				}
				else{
					// cout<<"Data is not in memory, pulling from DB\n";
					curridxpoint = item;
		 			// leveldb::Options options;
					// leveldb::DB* db;
					// leveldb::Status s = leveldb::DB::Open(options, featureFile, &db);
					stringstream ss;
					ss << setw(10) << setfill('0') << item+1;
					string key = ss.str();
					string value;
					leveldb::Status s = db->Get(leveldb::ReadOptions(), key, &value);
					currfeature = SparseFeature();
					currfeature.num_uniq_wrds = line2Features(value, currfeature, nFeatures);
					// delete db;
				    //cout<<"Finished Data is out of memory, accessing from DB\n";
					return currfeature;
				}
			}
		}
	}
	
	SparseFeature& SparseRepresentation::operator[](int item){
		return getFeature(item);
	}
    
    void writeFeaturesLevelDBBinary(char* outputlevelDBFile, char* inputlevelDBFile, int mode, int verbosity){
        if (verbosity > 0){
            printf("Reading feature file from %s...\n", inputlevelDBFile);
            printf("Output levelDB file into %s...\n", outputlevelDBFile);
        }
        leveldb::DB* db_local;
        leveldb::Options options_local;
        leveldb::DB* db_write;
        leveldb::Options options_write;
        options_write.create_if_missing = true;
        options_write.error_if_exists = false;
        options_write.write_buffer_size = 268435456;
        leveldb::Status s_write = leveldb::DB::Open(options_write, outputlevelDBFile, &db_write);
        leveldb::Status s = leveldb::DB::Open(options_local,inputlevelDBFile,&db_local);
        const int kMaxKeyLength = 256;
        char key_cstr[kMaxKeyLength];

        if (!s.ok()){
            cerr<<"DB error:"<<s.ToString()<<"\n";
            exit (-1);
        }
        if (!s_write.ok()){
            cerr<<"DB error:" <<s_write.ToString() << endl;
            exit (-1);
        }

        string key;
        string line;

        leveldb::Iterator* it = db_local->NewIterator(leveldb::ReadOptions());
        int lineno = 0;
        int a, b;
        int nFeatures;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            //cout << "Key : " << key;
            //cout << " String: " << line << endl;
            key = it->key().ToString();
            line = it->value().ToString();
            if (lineno == 0){
                //cout << "Key : " << key << endl;
                //cout << "String: " << line << endl;
                int pos = 0;
                sscanf(line.c_str(),"%d %d %n",&a,&b, &pos);
                cout<<"n = "<<a<<" and "<< "numFeatures = "<<b<<"\n";
                leveldb::WriteBatch batch;
                stringstream ss;
                //string id = "InitialKey";
                batch.Put(key, line);
                s_write = db_write->Write(leveldb::WriteOptions(), &batch);

                lineno++;
            }
            else{
                //features.push_back(Feature());     
                cout << "Now processing line " << lineno << endl;
                //cout << "Key : " << key << endl;
                //cout << "String: " << line << endl;
                struct SparseFeature feature_item;
                feature_item.index = lineno;
                feature_item.num_uniq_wrds = char2Features(line.c_str(), feature_item, nFeatures, 0);
                leveldb::WriteBatch batch;
                stringstream ss;
                ss << setw(10) << setfill('0') << (lineno);
                string id = ss.str();
                std::string out_str;
                if (mode == 1){ // float binary
                    float* vec = new float[feature_item.num_uniq_wrds*2];
                    for(int idx = 0; idx<feature_item.num_uniq_wrds; idx++){
                        vec[idx*2] = (float) feature_item.featureIndex[idx];
                        vec[idx*2+1] = (float) feature_item.featureVec[idx];
                    }
                    const char* char_str = (const char*) vec;
                    out_str = std::string(char_str, feature_item.num_uniq_wrds*2*sizeof(float)/sizeof(char));
                    
                    delete [] vec;
                }
                else if (mode == 2){ // double binary
                    double* vec = new double[feature_item.num_uniq_wrds*2];
                    for(int idx = 0; idx < feature_item.num_uniq_wrds;idx++){
                        vec[idx*2] = (double) feature_item.featureIndex[idx];
                        vec[idx*2+1] = (double) feature_item.featureVec[idx];
                    }
                    const char* char_str = (const char*) vec;
                    out_str = std::string(char_str, feature_item.num_uniq_wrds*2*sizeof(double)/sizeof(char));
                    delete [] vec;
                }
                batch.Put(id , out_str);
                s_write = db_write->Write(leveldb::WriteOptions(), &batch);

                lineno ++;
            }	
        }	
        delete db_local;
        delete db_write;
    }
		
}
