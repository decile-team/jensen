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
#include "time.h"
using namespace std;

#include "SparseRepresentation.h"
using namespace jensen;

int main()
{
    char* leveldb = "leveldbFile";
	time_t start, end;
	double timetaken;
#if 0
	SparseRepresentation fP (leveldb, 2);
	for(int i = 0; i < fP[200].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[200].featureIndex[i]<<" and Value: "<<fP[200].featureVec[i]<<"\n";
	}
#endif
#if 0
	SparseRepresentation fP (textfile, 1);
	/*for(int i = 0; i < fP[200].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[200].featureIndex[i]<<" and Value: "<<fP[200].featureVec[i]<<"\n";
	}*/
#endif
#if 1
	int batchsize = 500000;
	SparseRepresentation fP(leveldb, 2, true);
	vector<int> cset = vector<int>();
	for (int i = 0; i < batchsize; i++)
	{
		cset.push_back(i);
	}
	start = time(0);
	// fP.resetRepresentationSet(cset);
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken to reset representation is "<<timetaken<<" seconds.\n";
	start = time(0);
	for(int i = 0; i < fP[2000].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[2000].featureIndex[i]<<" and Value: "<<fP[2000].featureVec[i]<<"\n";
	}
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken for in memory read is "<<timetaken<<" seconds.\n";
	cout<<"\n";
	start = time(0);
	for(int i = 0; i < fP[712833].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[712833].featureIndex[i]<<" and Value: "<<fP[712833].featureVec[i]<<"\n";
	}
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken for out of memory read is "<<timetaken<<" seconds.\n";
	cset.clear();
	cout<<"\n";
	
    /*
    start = time(0);
	fP.resetRepresentationSequential(batchsize, batchsize, cset);
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken to reset representation is "<<timetaken<<" seconds.\n";
	for(int i = 0; i < fP[572334].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[572334].featureIndex[i]<<" and Value: "<<fP[572334].featureVec[i]<<"\n";
	}	
	cout<<"\n";
	for(int i = 0; i < fP[24500].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[24500].featureIndex[i]<<" and Value: "<<fP[24500].featureVec[i]<<"\n";
	}
	cout<<"\n";
	unordered_set<int> randomSet = unordered_set<int>();
	start = time(0);
	fP.resetRepresentationRandom(cset, batchsize, randomSet);
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken to reset representation is "<<timetaken<<" seconds.\n";
	unordered_set<int>::iterator it = randomSet.begin();
	cout<<*it<<"\n";
	start = time(0);
	for(int i = 0; i < fP[*it].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[*it].featureIndex[i]<<" and Value: "<<fP[*it].featureVec[i]<<"\n";
	}	
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken to in memory data access is "<<timetaken<<" seconds.\n";
	cout<<"\n";
	start = time(0);
	if (randomSet.find(30000) == randomSet.end())
		cout<<"The item 30000 does not belong to randomset\n";
	for(int i = 0; i < fP[30000].num_uniq_wrds; i++)
	{
		cout<<"Idx: "<<fP[30000].featureIndex[i]<<" and Value: "<<fP[30000].featureVec[i]<<"\n";
	}
	end = time(0);
	timetaken = difftime(end, start);
        cout<<"The time taken for out of memory memory data access is "<<timetaken<<" seconds.\n";
	cout<<"\n";
	*/
#endif
}
