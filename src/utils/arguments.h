/*-
 *-----------------------------------------------------------------------
 * A simple library to parse command lines in C++ using
 * an easy interface to quickly define arguments.
 *
 *       Jeff Bilmes <bilmes@ee.washington.edu>

   Modified by Karim Filali (karim@cs.washington.edu) to handle the following:

   - "Array type" flags: an example is the input flag -i; before if we
   wanted to have five input files we had to have five different flag
   entries -i1,-i2,...,-i5, all of which have the same properties.
   Now we only need to specify a generic flag -iX which is tied to an
   array instead to just a variable. -i1 corresponds to the first
   element of the array and so on.
   There are a few options relating to how to treat potentially ambiguous flags:

     - The variable "static bool Disambiguate_When_Perfect_Match"
     controls whether we want to allow switches such as -i, -invert,
     and -input. If the bool below is false, command line flags -i and
     -in are both ambiguous. If it is true, -i is no longer ambiguous
     because it mathes the arg -i perfectly.

     - The variable "static bool
     Disambiguate_When_Different_Data_Struct_Type" controls whether we
     want to allow an array command line arg -in1 when we have say
     switches -invert (SINGLE switch), and -input (ARRAY).  If the
     bool below is true, -in1 is valid since it is a array type switch
     (assuming no -in1 arg exits) and -input is the only matching
     template of the same type.  If the bool is false, we don't
     disambiguate

     - The variable "static bool
     Error_If_No_Index_For_Array_Data_Struct_Type", if true, cause an
     error if the user does not specify the index (e.g. -i instead
     of -i1) for array type flags. If false -i is equivalent to -i1.
   (2003/08/15)

   - Ability to hide some flags and to selectively print usage
   information about a subset of the flags based on the prefix string.
   For example the flag -klt has several suboptions, -kltUnityVar,
   -kltInputStat, etc, that are only relevant when the main flag is
   used.  By making that flag hidden, usage information about the
   subflags will not be printed unless we specify the option "-usage
   klt" The way the suboptions are linked to a min option is by
   sharing the same prefix, "klt" in this case.
   (2003/12/22)

   Also, I changed a few things such as the handling of argument errors:
   now we do not exit when such an error occurs but return an error code
   so that the calling routine gets a chance to print specific usage
   information before quiting.

   I also changed the names of the C style include files to get rid of
   warnings.  There are still a few warnings left though.

   Implementation-wise I replaced the previous MultiArg union mechanism
   by one in which I just use a void pointer and cast appropriately.  I
   needed to do that because the MultiArg union could not handle array
   type variables.

   TODO: cleanup; a better driver program.

 *
 *  $Header: /homes/bilmes/.cvsroot/gmtk_dev/miscSupport/arguments.h,v 1.10 2007/08/20 09:11:15 bilmes Exp $
 *
 *-----------------------------------------------------------------------
 */


#ifndef ARGS_h
#define ARGS_h

#include <iostream>
#include <cstdio>

#define DEFAULT_MAX_NUM_ARRAY_ELEMENTS 1
enum Priorities {
	HIGHEST_PRIORITY=1,
	PRIORITY_2,
	PRIORITY_3,
	PRIORITY_4,
	LOWEST_PRIORITY
};

class MultiType {
friend class Arg;
enum ArgumentType {
	int_type, // integer
	uint_type, // unsigned integer
	float_type, // single precision float
	double_type, // double precision float
	str_type, // string
	char_type, // char
	bool_type // boolean
};

void * ptr;
ArgumentType type;
static const char* printable(MultiType::ArgumentType);
public:
MultiType(bool& b) {
	ptr = (void*)&b; type = bool_type;
}
MultiType(char& c) {
	ptr = (void*)&c;  type = char_type;
}
MultiType(char*& s) {
	ptr = (void*)&s; type = str_type;
}
MultiType(const char*& s) {
	ptr = (void*)&s; type = str_type;
}
MultiType(int& i)  {
	ptr = (void*)&i;  type = int_type;
}
MultiType(unsigned int& i)  {
	ptr = (void*)&i; type = uint_type;
}
MultiType(float& f) {
	ptr = (void*)&f; type = float_type;
}
MultiType(double& d) {
	ptr = (void*)&d; type = double_type;
}

MultiType(char** s) {
	ptr = (void*)s; type = str_type;
}
MultiType(const char** s) {
	ptr = (void*)s; type = str_type;
}
MultiType(bool* b) {
	ptr = (void*)b; type = bool_type;
}
MultiType(int* i)  {
	ptr = (void*)i; type = int_type;
}
MultiType(unsigned int* i)  {
	ptr = (void*)i;  type = uint_type;
}
MultiType(float* f) {
	ptr = (void*)f;  type = float_type;
}
MultiType(double* d) {
	ptr = (void*)d;  type = double_type;
}


void print(FILE*);
};


class Arg {
friend class MultiType;
public:

// the argument disposition, optional, required, or toggle
enum ArgDisposition { Opt, Req, Tog, Help };
// the return codes, missing, ok, or in error.
enum ArgsRetCode { ARG_MISSING, ARG_OK, ARG_ERROR };
// the argument data structure type: single variable or array.
enum ArgDataStruct { SINGLE, ARRAY };
// the actual argument array itself.
static Arg Args[];

private:

/////////////////////////////////////////////////////////
// Static members shared between all arumnet instances
/////////////////////////////////////////////////////////

// This specifies whether we want to allow switches such as -i, -invert, and -input.
// If the bool below is false, command line flags -i and -in are both ambiguous.
// If it is true, -i is no longer ambiguous because it mathes the arg -i perfectly.
static bool Disambiguate_When_Perfect_Match;

// This specifies whether we want to allow an array command line arg
// -in1 when we have say switches -invert (SINGLE switch), and
// -input (ARRAY).  If the bool below is true, -in1 is valid since
// it is a array type switch (assuming no -in1 arg exits) and -input
// is the only matching template of the same type.  If the bool is
// false, we don't disambiguate
static bool Disambiguate_When_Different_Data_Struct_Type;

// If true, cause an error if the user does not specify the index (e.g. -i
// instead of -i1) for array type flags. If false -i is equivalent to -i1.
static bool Error_If_No_Index_For_Array_Data_Struct_Type;

static const char* const NOFLAG;
static const char* const CATEG_FLAG;
static const char* const NOFL_FOUND;
static const char COMMENTCHAR;    // for argument files.
// the total number of arguments in a given program.
static int Num_Arguments;
// an array of bits set if a particular argument is used
static bool* Argument_Specified;
// the program name, saved for usage messages.
static const char* Program_Name;

// The priority level at which the user wants the information
// printed.  Arguments with priority greater or equal to the
// requesteted priority level will be printed. Other argumnets will
// not.
static unsigned Requested_Priority;

//////////////////////////////
// Instance data members.
//////////////////////////////

const char *flag;   // name to match on command line, NULL when end.
ArgDisposition arg_kind;    // optional, required, toggle
MultiType mt;
const char *description;

ArgDataStruct dataStructType;
int arrayElmtIndex;
unsigned maxArrayElmts;

bool hidden;    // if true makes this flag not appear in the usual usage message.

// special lvalue boolean to be able to construct empty argument
// entry for end of array.
static bool EMPTY_ARGS_FLAG;

// determines if this argument gets printed in the usage information
// if a user provided priority number is higher than the below.
unsigned priority;

char* category;

unsigned count;    // keeps track of how many command line instances of this flag there are.

public:

// constructors.
Arg();
Arg(const char*d);
Arg(const char*,ArgDisposition,MultiType,const char*d=NULL,ArgDataStruct ds=SINGLE,
    unsigned maxArrayElmts=DEFAULT_MAX_NUM_ARRAY_ELEMENTS, bool hidden=false,
    unsigned priority=HIGHEST_PRIORITY);
Arg(ArgDisposition,MultiType,const char*d=NULL,ArgDataStruct ds=SINGLE,
    unsigned maxArrayElmts=DEFAULT_MAX_NUM_ARRAY_ELEMENTS, bool hidden=false,
    unsigned priority=HIGHEST_PRIORITY);
Arg(const Arg&);


//Arg(char*,ArgDisposition,MultiType,char*d=NULL);
//Arg(ArgDisposition,MultiType,char*d=NULL);


~Arg();

static ArgsRetCode parseArgsFromCommandLine(int,char**);
static ArgsRetCode parseArgsFromFile(char*f="argsFile");
static bool parse(int i,char**c);
static void usage(const char* filter=NULL, bool stdErrPrint = true);
static void printArgs(Arg*args,FILE*f);

static unsigned getNumArguments() {
	return (unsigned) Num_Arguments;
}
static unsigned getNumSuppliedArguments() {
	unsigned cnt=0;
	for(int i=0; i< Num_Arguments; ++i) {
		if(Argument_Specified[i]) cnt++;
	}
	return cnt;
}

private:
//void initialize(char*,ArgDisposition,char*);
void initialize(const char*,ArgDisposition,const char*,ArgDataStruct,unsigned,bool,unsigned priority);

static bool noFlagP(const char *);
static bool categFlagP(const char *);
static ArgsRetCode argsSwitch(Arg*,const char *,int&,bool&,const char*);
static Arg* searchArgs(Arg*,const char*);
static Arg* searchArgs(Arg* ag,char *flag, ArgDataStruct dataStructure);
static Arg* searchArrayArgs(Arg* Args, char* flag);
static void countAndClearArgBits();
static bool checkMissing(bool printMessage=false);
static bool validBoolean(const char* string,bool&value);
void print(FILE*);

void incCount() {
	this->count++;
}
unsigned getCount() {
	return this->count;
}

unsigned getPriority() {
	return this->priority;
}

};


#endif
