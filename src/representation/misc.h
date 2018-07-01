// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef SMTK_MISC
#define SMTK_MISC

namespace jensen{

int str2int(const char *p) {
    int x = 0;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    while (*p >= '0' && *p <= '9') {
        x = (x*10) + (*p - '0');
        ++p;
    }
    if (neg) {
        x = -x;
    }
    return x;
}

double str2double(const char *p) {
    double r = 0.0;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    while (*p >= '0' && *p <= '9') {
        r = (r*10.0) + (*p - '0');
        ++p;
    }
    if (*p == '.') {
        double f = 0.0;
        int n = 0;
        ++p;
        while (*p >= '0' && *p <= '9') {
            f = (f*10.0) + (*p - '0');
            ++p;
            ++n;
        }
        r += f / std::pow(10.0, n);
    }
    if (neg) {
        r = -r;
    }
    return r;
}
int line2FeaturesDouble(string line, struct SparseFeature & feature, int nFeatures, bool safe){

    int n_char = line.size();
    int n_wrds = n_char / sizeof(double);
    if ((n_wrds%2) != 0){
        cout << line << endl;
        cerr << "The above string has something wrong, please double check " << endl;
    }
    const char* char_ptr = line.c_str();
    double* out_vec = (double*) char_ptr;
    for (int idx=0;idx<n_wrds/2;idx++){
        feature.featureIndex.push_back((int)out_vec[2*idx]); // cast the type from double to int
        feature.featureVec.push_back((float)out_vec[2*idx+1]); // case the type from double to float
        //cout << out_vec[2*idx] << " " << out_vec[2*idx+1] << " ";
    }
    //cout << endl;
    return (n_wrds/2);
}

int line2FeaturesFloat(string line, struct SparseFeature & feature, int nFeatures, bool safe){

    int n_char = line.size();
    int n_wrds = n_char / sizeof(float);
    if ((n_wrds%2) != 0){
        cout << line << endl;
        cerr << "The above string has something wrong, please double check " << endl;
    }
    const char* char_ptr = line.c_str();
    float* out_vec = (float*) char_ptr;
    for (int idx=0;idx<n_wrds/2;idx++){
        feature.featureIndex.push_back((int)out_vec[2*idx]);
        feature.featureVec.push_back((float)out_vec[2*idx+1]);
        //cout << out_vec[2*idx] << " " << out_vec[2*idx+1] << " ";
    }
    //cout << endl;
    return (n_wrds/2);
}


int string2Features(string line, struct SparseFeature & Feature, int nFeatures, bool safe){
	int digitwrd;
	float featureval;
	int pos = 0;
	int num_words = 0;
	stringstream stream(line);
	while (stream>>digitwrd){
		if(! (stream>>featureval))
			cerr<<"Mismatch in the input data provided. Please check the input data\n";
	 	num_words++;
		if (safe == 1){
			if ((digitwrd >= nFeatures) || (digitwrd < 0)){
		    		cout << "The input feature graph is not right: " << " some feature index is either >= nFeatures or <0. It has a value of "<< digitwrd<<". Please make sure the feature indices ranging from 0 to nFeatures-1\n";
			}
			assert((digitwrd < nFeatures) && (digitwrd >= 0));
	       		 if (featureval < 0){
		    		cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
			}
			assert(featureval >= 0);
		}
		Feature.featureIndex.push_back(digitwrd);
		Feature.featureVec.push_back(featureval);
	}
	return num_words;
}

int string2FeaturesFast(string line, struct SparseFeature & Feature, int nFeatures, bool safe){
	int digitwrd;
	int num_words = 0;
	float featureval;
	size_t start=0;
    	size_t end=line.find_first_of(' ');
	while (end <= std::string::npos){
		digitwrd = str2int(line.substr(start, end-start).c_str());
		start=end+1;
    		end = line.find_first_of(' ', start);
		if (end == std::string::npos){
			cerr<<"Mismatch in the input data provided. Please check the input data\n";
			return 0;
		}
		featureval = str2double(line.substr(start, end-start).c_str());
	 	num_words++;	
		if (safe == 1){
			if ((digitwrd >= nFeatures) || (digitwrd < 0)){
		    		cout << "The input feature graph is not right: " << " some feature index is either >= nFeatures or <0. It has a value of "<< digitwrd<<". Please make sure the feature indices ranging from 0 to nFeatures-1\n";
			}
			assert((digitwrd < nFeatures) && (digitwrd >= 0));
	       		 if (featureval < 0){
		    		cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
			}
			assert(featureval >= 0);
		}
		Feature.featureIndex.push_back(digitwrd);
		Feature.featureVec.push_back(featureval);
		start=end+1;
    		end = line.find_first_of(' ', start);
		if (end == std::string::npos){
			break;
		}
	}
	return num_words;
}

int char2Features(const char *s, struct SparseFeature & Feature, int nFeatures, bool safe){
	int digitwrd;
	float featureval;
	int pos = 0;
	int num_words = 0;
	while (sscanf(s,"%d %f %n",&digitwrd, &featureval, &pos) == 2) {
	 s += pos;
	 num_words++;
	if (safe == 1){
		if ((digitwrd >= nFeatures) || (digitwrd < 0)){
		    cout << "The input feature graph is not right: " << " some feature index is either >= nFeatures or <0, please make sure the feature indices ranging from 0 to nFeatures-1\n";
            cout << digitwrd << endl;
		}
		assert((digitwrd < nFeatures) && (digitwrd >= 0));
		if (featureval < 0){
		    cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
		}
		assert(featureval >= 0);
	}
	Feature.featureIndex.push_back(digitwrd);
	Feature.featureVec.push_back(featureval);
	}
	return num_words;
}



}
#endif
