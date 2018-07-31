/*-
 *-----------------------------------------------------------------------
 * A simple library to parse command lines in C++ using
 * an easy interface to quickly define arguments.
 *
 *   Jeff Bilmes <bilmes@ee.washington.edu>

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
 *  $Header: /homes/bilmes/.cvsroot/gmtk_dev/miscSupport/arguments.cc,v 1.18 2007/08/20 09:11:15 bilmes Exp $
 *
 *-----------------------------------------------------------------------
 */

using namespace std;

#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <assert.h>


#include "error.h"
#include "arguments.h"


/*-
 *-----------------------------------------------------------------------
 * static entities
 *-----------------------------------------------------------------------
 */

bool Arg::Disambiguate_When_Perfect_Match=true;
bool Arg::Disambiguate_When_Different_Data_Struct_Type=true;
bool Arg::Error_If_No_Index_For_Array_Data_Struct_Type=true;

bool Arg::EMPTY_ARGS_FLAG;

const char* const Arg::NOFLAG = "noflg";
// special flag that indicates the argument is for printing category heading information only
const char* const Arg::CATEG_FLAG = "categflg";
const char* const Arg::NOFL_FOUND = "nofl_fnd";
const char Arg::COMMENTCHAR = '#';
const char* const ARGS_FILE_NAME = "argsFile";
const char* const ArgsErrStr = "Argument Error:";
const char* const ArgsWarningStr = "Argument Warning:";


int Arg::Num_Arguments = 0;

bool* Arg::Argument_Specified = NULL;
const char* Arg::Program_Name = "";
unsigned Arg::Requested_Priority=HIGHEST_PRIORITY;

/*-
 *-----------------------------------------------------------------------
 * CONSTRUCTORS
 *-----------------------------------------------------------------------
 */

/*-
 *-----------------------------------------------------------------------
 * Arg::initialize
 *      initialize the structures.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void Arg::initialize(const char*m,
                     ArgDisposition r,
                     const char *d,
                     ArgDataStruct ds,
                     unsigned maxArrayElmts,
                     bool hidden,
                     unsigned priority) {
	flag = m;
	arg_kind = r;
	if (arg_kind == Tog) {
		// type must be boolean. Toggle is never a required arg.
		if (mt.type != MultiType::bool_type) {
			error("%s A toggle argument must boolean",ArgsErrStr);
		}
	}
	if(arg_kind == Help) {
		// cannot be of array type.
		if(ds == ARRAY) {
			error("INTERNAL ERROR: Cannot have a flag of kind Help that is also of type ARRAY.");
		}
	}
	description = d;
	dataStructType=ds;
	this->maxArrayElmts=maxArrayElmts;
	arrayElmtIndex=0;
	this->hidden = hidden;
	this->priority=priority;
	this->count=0;
}


/*-
 *-----------------------------------------------------------------------
 * Arg::Arg()
 *      creates a special empty argument structure
 *      for using at the end of a list.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::Arg() : mt(EMPTY_ARGS_FLAG) {
	initialize(NULL,
	           Arg::Opt,
	           "",
	           SINGLE,
	           DEFAULT_MAX_NUM_ARRAY_ELEMENTS,
	           false,
	           HIGHEST_PRIORITY);
}

/**
 *-----------------------------------------------------------------------
 * Arg::Arg()
 *      creates a special args oject to print category information
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 **/
Arg::Arg(const char* d) : mt(EMPTY_ARGS_FLAG) {
	initialize(CATEG_FLAG,Arg::Opt,d,SINGLE,DEFAULT_MAX_NUM_ARRAY_ELEMENTS,false,HIGHEST_PRIORITY);
}


/*-
 *-----------------------------------------------------------------------
 * Arg::Arg()
 *      creates a complete args object, with all bells and whistles.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::Arg(const char*m,ArgDisposition r,MultiType ucl,const char* d,ArgDataStruct ds,unsigned maxArrayElmts, bool hidden, unsigned priority) : mt(ucl) {
	initialize(m,r,d,ds,maxArrayElmts,hidden,priority);
}


/*-
 *-----------------------------------------------------------------------
 * Arg::Arg()
 *      creates a flagless argument.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::Arg(ArgDisposition r,MultiType ucl,const char* d,ArgDataStruct ds,unsigned maxArrayElmts,bool hidden,unsigned priority) : mt(ucl) {
	//initialize(NOFLAG,r,d);
	initialize(NOFLAG,r,d,ds,maxArrayElmts,hidden,priority);
}


/*-
 *-----------------------------------------------------------------------
 * Arg::Arg()
 *      copy constructor.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::Arg(const Arg& a)
	: mt(a.mt) {
	flag = a.flag;
	arg_kind = a.arg_kind;
	description = a.description;
}

Arg::~Arg() {
	if(Argument_Specified != NULL) {
		delete [] Argument_Specified;
		Argument_Specified=NULL;
	}
}


/**
 *-----------------------------------------------------------------------
 *  searchArrayArgs()
 *
 *   remove int postfix of flag, if any, and search the arg list one
 *   that matches a given flag.
 *
 * Preconditions:
 *
 *      should be called after a call to searchArgs() has failed.  No
 *      catastropchic consequecences otherwise but you might get a
 *      ambiguous switch return value.
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      Return NULL for unknown switch, return (Arg*)(-1) for ambiguous switch,
 *      otherwise returns the Arg* that matches.
 *
 *----------------------------------------------------------------------- */
Arg* Arg::searchArrayArgs(Arg* Args, char* flag) {
	int flaglen = ::strlen(flag);
	char* flagCopy=::strcpy(new char[flaglen+1],flag);
	char* flagPtr=&flagCopy[flaglen-1];
	int index;

	Arg* arg_ptr=NULL;

	if( isdigit(*flagPtr) ) { // check last character of flag
		while( isdigit(*flagPtr) ) flagPtr--;
		index = atoi(++flagPtr);
		*flagPtr='\0';
		if(Disambiguate_When_Different_Data_Struct_Type)
			arg_ptr = searchArgs(Args,&flagCopy[1],ARRAY);
		else
			arg_ptr = searchArgs(Args,&flagCopy[1]);
		if(arg_ptr==(Arg*)(-1)) {
			warning("%s Ambiguous switch: %s",ArgsErrStr,flag);
			return (Arg*)(-1);
		}
		else if(arg_ptr != NULL) { // this time we found it
			arg_ptr->arrayElmtIndex= index-1; // The user number flags starting from 1.
			if(arg_ptr->arrayElmtIndex > (int) arg_ptr->maxArrayElmts-1) {
				warning("%s Array index in switch (%s) is out of bounds (1..%d)",
				        ArgsErrStr, flag,arg_ptr->maxArrayElmts);
				return (Arg*)(-1);
			}
			// if the argument type of the matching argument is not ARRAY then we should stop here;  otherwise things will break down later on.
			if(arg_ptr->dataStructType != ARRAY) {
				warning("%s Switch (%s) is not of type ARRAY and cannot be written as (%sX)", ArgsErrStr,flag,flagCopy);
				return (Arg*)(-1);
			}
		}
	}

	return arg_ptr;
}

/*-
 *-----------------------------------------------------------------------
 * countAndClearArgBits()
 *      count the number of arguments that are used, and clear any of the
 *      used bits (if this hasn't been called before)
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void Arg::countAndClearArgBits() {
	if (Argument_Specified != NULL)
		return;
	Num_Arguments = 0;
	Arg* arg_ptr = Args;
	while (arg_ptr->flag != NULL) {
		Num_Arguments++;
		arg_ptr++;
	}
	Argument_Specified = new bool[Num_Arguments];
	for (int i=0; i<Num_Arguments; i++) {
		Argument_Specified[i] = false;
	}
}


/*-
 *-----------------------------------------------------------------------
 * checkMissing()
 *      check if any required arguments are missing.
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      return true if any are missing.
 *
 *-----------------------------------------------------------------------
 */
bool Arg::checkMissing(bool printMessage) {
	Arg* arg_ptr;
	arg_ptr = Args;
	bool missing = false;
	while (arg_ptr->flag != NULL) {
		if (!Argument_Specified[arg_ptr-Args] &&
		    arg_ptr->arg_kind == Req) {
			if (printMessage) {
				char brackets[2];
				fprintf(stderr,"%s Missing REQUIRED argument:",ArgsErrStr);
				if (!noFlagP(arg_ptr->flag)) {
					if(arg_ptr->dataStructType == ARRAY)
						fprintf(stderr," -%sX ",arg_ptr->flag);
					else
						fprintf(stderr," -%s ",arg_ptr->flag);
				}
				// bool_type args are always optional. (i.e. -b T, -b F, or -b)
				if (arg_ptr->mt.type == MultiType::bool_type) {
					brackets[0] = '[';  brackets[1] = ']';
				} else {
					brackets[0] = '<'; brackets[1] = '>';
				}
				fprintf(stderr," %c%s%c\n",
				        brackets[0],
				        arg_ptr->mt.printable(arg_ptr->mt.type),
				        brackets[1]);
			}
			missing = true;
		}
		arg_ptr++;
	}
	return missing;
}


/*-
 *-----------------------------------------------------------------------
 * searchArgs()
 *      search the arg list one that matches a given flag.
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      Return NULL for unknown switch, return (Arg*)(-1) for ambiguous switch,
 *      otherwise returns the Arg* that matches.
 *
 *-----------------------------------------------------------------------
 */
Arg* Arg::searchArgs(Arg* ag,const char *flag) {
	int flaglen = ::strlen(flag);
	Arg* arg_ptr = ag;
	int numTaged = 0;
	int lastTaged = -1;
	int lastPerfectlyTaged = -1;
	bool perfectMatch=false;

	// find the one that best matches.
	while (arg_ptr->flag != NULL) {
		if (!noFlagP(arg_ptr->flag)) {
			if (!::strncmp(arg_ptr->flag,flag,flaglen)) {
				numTaged++;
				lastTaged = (arg_ptr - ag);
				if(::strlen(arg_ptr->flag)==(unsigned)flaglen) {
					perfectMatch=true;
					lastPerfectlyTaged=(arg_ptr - ag);
				}
			}
		}
		arg_ptr++;
	}
	// include check for args file since we shouldn't
	// be ambiguous with respect to this as well.
	if (!::strncmp(ARGS_FILE_NAME,flag,flaglen))
		numTaged++;

	if (numTaged == 0)
		return NULL;
	else if (numTaged > 1) {
		if(Disambiguate_When_Perfect_Match && perfectMatch) {
			return &ag[lastPerfectlyTaged];
		}
		return (Arg*)(-1);
	}
	else
		return &ag[lastTaged];
}


/*-
 *-----------------------------------------------------------------------
 * overloaded searchArgs()
 *
 *      search the arg list one that matches a given flag given that
 *      the args are of a certain type (SINGLE or ARRAY).
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      Return NULL for unknown switch, return (Arg*)(-1) for ambiguous switch,
 *      otherwise returns the Arg* that matches.
 *
 *----------------------------------------------------------------------- */
Arg* Arg::searchArgs(Arg* ag,char *flag, ArgDataStruct dataStructure) {
	int flaglen = ::strlen(flag);
	Arg* arg_ptr = ag;
	int numTaged = 0;
	int lastTaged = -1;
	int lastPerfectlyTaged = -1;
	bool perfectMatch=false;

	// find the one that best matches.
	while (arg_ptr->flag != NULL) {
		if (!noFlagP(arg_ptr->flag) && arg_ptr->dataStructType == dataStructure) {
			if (!::strncmp(arg_ptr->flag,flag,flaglen)) {
				numTaged++;
				lastTaged = (arg_ptr - ag);
				if(::strlen(arg_ptr->flag)==(unsigned)flaglen) {
					perfectMatch=true;
					lastPerfectlyTaged=(arg_ptr - ag);
				}
			}
		}
		arg_ptr++;
	}
	// include check for args file since we shouldn't
	// be ambiguous with respect to this as well.
	if (!::strncmp(ARGS_FILE_NAME,flag,flaglen) && dataStructure == SINGLE)
		numTaged++;

	if (numTaged == 0)
		return NULL;
	else if (numTaged > 1) {
		if(Disambiguate_When_Perfect_Match && perfectMatch) {
			return &ag[lastPerfectlyTaged];
		}
		return (Arg*)(-1);
	}
	else
		return &ag[lastTaged];
}

/*-
 *-----------------------------------------------------------------------
 * argsSwitch()
 *      checks the argument to see if it has the appropriate value
 *      for a given flag (which already was found out).
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      only returns Arg::ARG_OK or ARG_ERROR
 *
 *-----------------------------------------------------------------------
 */
Arg::ArgsRetCode
Arg::argsSwitch(Arg* arg_ptr,const char *arg,int& index,bool& found,const char*flag)
{

	if(arg_ptr->dataStructType == ARRAY) {
		assert(arg_ptr->arrayElmtIndex >= 0 && arg_ptr->arrayElmtIndex < (int) arg_ptr->maxArrayElmts);
	}

	switch(arg_ptr->mt.type) {
	// =============================================================
	case MultiType::int_type: {
		if (arg == NULL) { // end of arguments.
			warning("%s Integer argument needed: %s",
			        ArgsErrStr,
			        flag);
			return ARG_ERROR;
		}
		int n;
		char *endp;
		n = (int)strtol(arg,&endp,0);
		if ( endp == arg ) {
			warning("%s Integer argument needed: %s %s",
			        ArgsErrStr,
			        flag,
			        arg);
			return ARG_ERROR;
		}
		if(arg_ptr->dataStructType == ARRAY)
			*((int*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = n;
		else
			*((int*)(arg_ptr->mt.ptr)) = n;
		found = true;
	}
	break;
	// =============================================================
	case MultiType::uint_type: {

		//////////////////////////////////////////////////////
		// Special case: help flag
		//////////////////////////////////////////////////////

		if(arg_ptr->arg_kind == Help) {
			if(arg==NULL || arg[0]=='-') { // end of arguments
				*((unsigned int*)(arg_ptr->mt.ptr)) = HIGHEST_PRIORITY;
				Requested_Priority = HIGHEST_PRIORITY;
			}
			else { // there is something after the switch and it is not another switch
				unsigned int n;
				char *endp;
				n = (unsigned)strtoul(arg,&endp,0);
				if ( endp != arg ) { // found an unsigned
					*((unsigned int*)(arg_ptr->mt.ptr)) = n;
					Requested_Priority = n;
				}
				// else no unsigned follows our switch of kind Help
				// therefore that must be flagless argument, which we ignore
			}
			found=true;
			break;
		}
		///////////////////////////////////////////////////////
		if (arg == NULL) { // end of arguments.
			warning("%s Unsigned integer argument needed: %s",
			        ArgsErrStr,
			        flag);
			return ARG_ERROR;
		}
		unsigned int n;
		char *endp;
		n = (unsigned)strtoul(arg,&endp,0);
		if ( endp == arg) {
			warning("%s Unsigned integer argument needed: %s %s",
			        ArgsErrStr,
			        flag,
			        arg);
			return ARG_ERROR;
		}
		if(arg_ptr->dataStructType == ARRAY)
			*((unsigned int*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = n;
		else
			*((unsigned int*)(arg_ptr->mt.ptr)) = n;
		found = true;
	}
	break;
	// =============================================================
	case MultiType::float_type: {
		if (arg == NULL) { // end of arguments.
			warning("%s Real number argument needed: %s",
			        ArgsErrStr,
			        flag);
			return ARG_ERROR;
		}
		float f;
		char *endp;
		f = strtof(arg,&endp);
		if ( arg == endp ) {
			warning("%s Floating point number argument needed: %s %s",
			        ArgsErrStr,
			        flag,
			        arg);
			return ARG_ERROR;
		}
		if(arg_ptr->dataStructType == ARRAY)
			*((float*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = f;
		else
			*((float*)(arg_ptr->mt.ptr)) = f;
		found = true;
	}
	break;
	// =============================================================
	case MultiType::double_type: {
		if (arg == NULL) { // end of arguments.
			warning("%s Real number argument needed: %s",
			        ArgsErrStr,
			        flag);
			return ARG_ERROR;
		}
		double d;
		char *endp;
		d = strtod(arg,&endp);
		if (arg == endp) {
			warning("%s Integer argument needed: %s %s",
			        ArgsErrStr,
			        flag,
			        arg);
			return ARG_ERROR;
		}
		if(arg_ptr->dataStructType == ARRAY)
			*((double*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = d;
		else
			*((double*)(arg_ptr->mt.ptr)) = d;
		found = true;
	}
	break;
	// =============================================================
	case MultiType::str_type: {
		if (arg == NULL) { // end of arguments.
			warning("%s String argument needed: %s",
			        ArgsErrStr,
			        flag);
			return ARG_ERROR;
		}
		// TODO: fix minor memory leak here, where old ptr is written over. Note that
		// it is not as simple as just freeing old pointer, as old pointer might point to
		// non-dynamicaly allocated memory (i.e., something like 'const char * foo = "bar"; ).
		if(arg_ptr->dataStructType == ARRAY)
			*( (char**)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = ::strcpy(new char[strlen(arg)+1],arg);
		else
			*((char**)(arg_ptr->mt.ptr)) = ::strcpy(new char[strlen(arg)+1],arg);
		found = true;
	}
	break;
	// =============================================================
	case MultiType::char_type: {
		if (arg == NULL || // end of arguments.
		    ::strlen(arg) != 1) {
			warning("%s Character argument needed: %s %s",
			        ArgsErrStr,
			        flag,arg);
			return ARG_ERROR;
		}
		if(arg_ptr->dataStructType == ARRAY)
			*((char*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex) = arg[0];
		else
			*((char*)(arg_ptr->mt.ptr)) = arg[0];
		found = true;
	}
	break;
	// =============================================================
	case MultiType::bool_type: {
		bool * argPtr;
		if(arg_ptr->dataStructType == ARRAY)
			argPtr = ((bool*)(arg_ptr->mt.ptr) + arg_ptr->arrayElmtIndex);
		else
			argPtr = (bool*)(arg_ptr->mt.ptr);
		bool b;
		if (arg == NULL || // end of arguments.
		    arg[0] == '-') { // for optionless flag, just turn on.
			// for bool case, just turn it on.
			if (arg_ptr->arg_kind != Tog) {
				*argPtr=true;
			}
			else
				*argPtr = ( *argPtr == true ) ? false : true;
			if (arg != NULL)
				index--;
			found = true;
			break;
		}
		// If kind  is Tog and we have a valid boolean
		// argument, then treat argument as normal explicit boolean argument.
		if (!validBoolean(arg,b)) {
			warning("%s Boolean argument needed: %s %s",
			        ArgsErrStr,
			        flag,arg);
			return ARG_ERROR;
		}
		*argPtr=b;
		found = true;
	}
	break;
	// =============================================================
	default:
		error("%s Unknown internal argument",ArgsErrStr);
		break;
	}
	return ARG_OK;
}


/*-
 *-----------------------------------------------------------------------
 * validBoolean()
 *      returns true (and its value) if string is a valid boolean value
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      boolean value
 *
 *-----------------------------------------------------------------------
 */
bool Arg::validBoolean(const char *string,bool& value)
{
	bool rc;
	int arglen = strlen(string);
	char *upcasearg = new char[arglen+1];
	::strcpy(upcasearg,string);
	for (int i=0; i<arglen; i++)
		upcasearg[i] = toupper(upcasearg[i]);
	if (!::strncmp("TRUE",upcasearg,arglen)
	    || !::strncmp("YES",upcasearg,arglen)
	    || !::strncmp("ON",upcasearg,arglen)
	    || (arglen == 1 && string[0] == '1'))  {
		value = true;
		rc = true;
	}
	else if (!::strncmp("FALSE",upcasearg,arglen)
	         || !::strncmp("NO",upcasearg,arglen)
	         || !::strncmp("OFF",upcasearg,arglen)
	         || (arglen == 1 && string[0] == '0')) {
		value = false;
		rc = true;
	} else {
		rc = false;
	}
	delete [] upcasearg;
	return rc;
}


// return true if there is no flag.
bool Arg::noFlagP(const char *flg) {
	// cant test here equality with NOFLAG, so do the following instread.
	if (flg == NOFLAG || flg == NOFL_FOUND)
		return true;
	else
		return false;
}

// return true if there is the falg is the special category flag.
bool Arg::categFlagP(const char *flg) {
	if (flg == CATEG_FLAG)
		return true;
	else
		return false;
}


/*-
 *-----------------------------------------------------------------------
 * Printing operations
 *-----------------------------------------------------------------------
 */


/*-
 *-----------------------------------------------------------------------
 *  MultiType::print(FILE* f)
 *      prints the argument value for the current arg.
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void MultiType::print(FILE* f) {
	switch (type) {
	case MultiType::bool_type:
		//fprintf(f,"%c",(ptr->boolean?'T':'F'));
		fprintf(f,"%c",(*((bool*)(ptr)) ? 'T' : 'F'));
		break;
	case MultiType::char_type:
		//fprintf(f,"%c",(ptr->ch));
		fprintf(f,"%c",*((char*)(ptr)));
		break;
	case MultiType::str_type:
		//fprintf(f,"%s",(ptr->string == NULL ? "" : ptr->string));
		fprintf(f,"%s",( ptr == NULL ? "" : *((char**)ptr)==NULL ? "null" : *((char**)ptr) ) );
		break;
	case MultiType::int_type:
		//fprintf(f,"%d",(ptr->integer));
		fprintf(f,"%d",*((int*)(ptr)));
		break;
	case MultiType::uint_type:
		//fprintf(f,"%u",(ptr->uinteger));
		fprintf(f,"%u",*((unsigned int*)(ptr)));
		break;
	case MultiType::float_type:
		//fprintf(f,"%e",(double)(ptr->single_prec));
		fprintf(f,"%e",(double)*((float*)(ptr)));
		break;
	case MultiType::double_type:
		//fprintf(f,"%e",(ptr->double_prec));
		fprintf(f,"%e",*((double*)(ptr)));
		break;
	default:
		error("%s Internal argument error",ArgsErrStr);
		break;
	}
}

/*-
 *-----------------------------------------------------------------------
 * Arg::printArgs()
 *      prints all the args in the list.

 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void Arg::printArgs(Arg*args,FILE* f) {
	Arg *arg_ptr = args;
	while (arg_ptr->flag != NULL) {
		arg_ptr->print(f);
		arg_ptr++;
	}
}

/*-
 *-----------------------------------------------------------------------
 * Arg::print()
 *      prints the current argument entry.

 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void Arg::print(FILE* f) {
	if (!noFlagP(flag))
		fprintf(f,"%s",flag);
	fprintf(f,":%s = ",MultiType::printable(mt.type));
	mt.print(f);
	fprintf(f,"; # %s\n",
	        ((description == NULL) ? "" : description));
}


/*-
 *-----------------------------------------------------------------------
 * printable()
 *      returns a printable version of an argument type
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      none
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
const char *MultiType::printable(MultiType::ArgumentType at) {
	switch (at) {
	case MultiType::bool_type:
		return "bool";
	case MultiType::char_type:
		return "char";
	case MultiType::str_type:
		return "str";
	case MultiType::int_type:
		return "integer";
	case MultiType::uint_type:
		return "unsigned";
	case MultiType::float_type:
		return "float";
	case MultiType::double_type:
		return "double";
	default:
		return "error: unknown type";
	}
}


/*-
 *-----------------------------------------------------------------------
 * usage()
 *      prints a usage message of the program. This will print out
 *      the arguments, their types, documentation, and default values.
 *
 * Preconditions:
 *      none
 *
 * Postconditions:
 *      stuff is printed out.
 *
 * Side Effects:
 *      none
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
void Arg::usage(const char* filter,bool stdErrPrint) {

	FILE* destStream;

	if (stdErrPrint)
		destStream = stderr;
	else
		destStream = stdout;

	fprintf(destStream,"Usage: %s  [[[-flag] [option]] ...]\n",Program_Name);
	fprintf(destStream,"Required: <>; Optional: []; Flagless arguments must be in order.\n");

	Arg* arg_ptr = Args;
	int longest_variation = 0;

	while (arg_ptr->flag != NULL) {
		int len = 0;
		if(categFlagP(arg_ptr->flag)) { arg_ptr++; continue;}
		if (!noFlagP(arg_ptr->flag)) {
			// add one for the '-', as in "-flag"
			len += ::strlen(arg_ptr->flag)+1;
			len++; //  add one for the ' ' in "-flag "
		}
		len += ::strlen(MultiType::printable(arg_ptr->mt.type));
		len += 2; // add two for brackets. '[',']', or '<','>' around type.
		if(arg_ptr->arg_kind==Help) // add [] around unsigned for the help flag
			len+=2;
		if (len  > longest_variation)
			longest_variation = len;
		arg_ptr++;
	}

	//  for (int printOptional=0;printOptional<2;printOptional++) {
	arg_ptr = Args;
	while (arg_ptr->flag != NULL) {
		int this_variation = 0;
		char brackets[2];
		if(arg_ptr->hidden && filter==NULL) goto skip;
		if(filter != NULL && ::strncmp(filter,arg_ptr->flag,strlen(filter))!=0) goto skip;
		if(arg_ptr->getPriority() > Requested_Priority) goto skip;

		if(categFlagP(arg_ptr->flag)) {
			fprintf(destStream,"%s\n",
			        ((arg_ptr->description == NULL) ? "" : arg_ptr->description));
			goto skip;
		}

		if (arg_ptr->arg_kind == Req) {
			//	if (printOptional) goto skip;
			brackets[0] = '<'; brackets[1] = '>';
		} else {
			//	if (!printOptional) goto skip;
			brackets[0] = '['; brackets[1] = ']';
		}
		fprintf(destStream," %c",brackets[0]);

		if (!noFlagP(arg_ptr->flag)) {
			// add one for the '-', as in "-flag"
			this_variation = ::strlen(arg_ptr->flag) + 1;
			if(arg_ptr->dataStructType==ARRAY) {
				fprintf(destStream,"-%sX",arg_ptr->flag);
				this_variation++;
			}
			else
				fprintf(destStream,"-%s",arg_ptr->flag);
			fprintf(destStream," ");
			this_variation++; //  add one for the ' ' in "-flag "
		}

		if(arg_ptr->arg_kind==Help)  {// add [] around unsigned for the help flag
			fprintf(destStream,"[%s]",arg_ptr->mt.printable(arg_ptr->mt.type));
			this_variation += 2; // account for the extra []
		}
		else {
			fprintf(destStream,"%s",arg_ptr->mt.printable(arg_ptr->mt.type));
		}
		this_variation += ::strlen(MultiType::printable(arg_ptr->mt.type));
		// add two for brackets. '[',']', or '<','>' around type.
		this_variation += 2;

		fprintf(destStream,"%c",brackets[1]);

		while (this_variation++ < longest_variation)
			fprintf(destStream," ");
		fprintf(destStream,"   %s {",
		        ((arg_ptr->description == NULL) ? "" : arg_ptr->description));
		arg_ptr->mt.print(destStream);
		fprintf(destStream,"}\n");

skip:
		arg_ptr++;
	}
	//}

	fprintf(destStream," [-%s <str>]",ARGS_FILE_NAME);
	int this_variation = 9 + strlen(ARGS_FILE_NAME);
	while (this_variation++ < longest_variation)
		fprintf(destStream," ");
	fprintf(destStream,"   File to obtain additional arguments from {}\n");
}


/*-
 *-----------------------------------------------------------------------
 * parseArgsFromFile()
 *      more direct parses from directly from a file
 *
 * Preconditions:
 *      file must be valid format.
 *
 * Postconditions:
 *      file is parsed, any previously defined argument values are lost.
 *
 * Side Effects:
 *      modifies internal object static variables.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::ArgsRetCode Arg::parseArgsFromFile(char *fileName)
{
	countAndClearArgBits();
	ifstream ifile(fileName);
	if (!ifile) {
		warning("%s Can't file argument file: %s",
		        ArgsErrStr,
		        fileName);
		return ARG_ERROR;
	} else {
		// get the remaining from file
		const unsigned max_line_length = 32*1024;
		char buffer[max_line_length];

		while (ifile.getline(buffer,max_line_length,'\n')) {
			// printf("Got line (%s)\n",buffer);
			const unsigned line_length = strlen(buffer);
			if (line_length+1 >= max_line_length) {
				// input string is too long.
				warning("%s Line length too long in command line parameter arguments file: %s",
				        ArgsErrStr,
				        fileName);
				return ARG_ERROR; // give up
			}
			char* buffp = buffer;
			while (*buffp) {
				if (*buffp == COMMENTCHAR) {
					*buffp = '\0';
					break;
				}
				buffp++;
			}

			buffp = buffer;
			while (*buffp && isspace(*buffp))
				buffp++; // skip space
			if (!*buffp) {
				continue; // empty line
			}
			char *flag = buffp; // get flag
			char *arg;
			// get command up to space or ':'
			while (*buffp && *buffp != ' ' && *buffp != '\t' &&  *buffp != '=')
				buffp++;
			if (buffp == flag)
				continue; // empty line or empty flag
			if (*buffp) {
				// get ':' and position buffp to start of arg.
				if (*buffp == '=')
					// we have the flag
					*buffp++ = '\0';
				else {
					// we have the flag, but need to get rid of ':' if there.
					*buffp++ = '\0';
					while (*buffp && *buffp != '=')
						buffp++;
					if (*buffp == '=')
						buffp++;
				}
				while (*buffp == ' ' || *buffp == '\t')
					buffp++; // skip space
			}

			if (!*buffp)
				arg = NULL;
			else {
				arg = buffp;
				// get command up to space
				while (*buffp && *buffp != ' ' && *buffp != '\t')
					buffp++;
				*buffp = '\0';
			}

			// check to see if it is the special parse from file
			// argument name, which is not valid here.
			if (!::strncmp(ARGS_FILE_NAME,flag,::strlen(flag)))
				error("%s Can not recursively parse arguments from files\n",ArgsErrStr);

			Arg* arg_ptr = searchArgs(Args,flag);
			if (arg_ptr == NULL) {
				warning("%s Skipping unknown switch in %s : %s,",
				        ArgsWarningStr,
				        fileName,flag);
				//return (ARG_ERROR);
			} else if (arg_ptr == (Arg*)(-1)) {
				warning("%s Ambiguous switch in %s : %s",
				        ArgsErrStr,
				        fileName,flag);
				return (ARG_ERROR);
			} else {
				int i;
				if (argsSwitch(arg_ptr,arg,i,Argument_Specified[arg_ptr - Args],flag)
				    != ARG_OK) {
					warning("%s Error in %s",
					        ArgsErrStr,
					        fileName);
					return (ARG_ERROR);
				}
			}
		}
	}
	if (checkMissing())
		return ARG_MISSING;
	return ARG_OK;
}



/*-
 *-----------------------------------------------------------------------
 * parseArgsFromCommandLine()
 *      more direct parses from the command line
 *
 * Preconditions:
 *      command line must be specified
 *
 * Postconditions:
 *      command line is parsed.
 *
 * Side Effects:
 *      modifies internal object static variables.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
Arg::ArgsRetCode
Arg::parseArgsFromCommandLine(int argc,char**argv)
{
	if (argv[0] != NULL)
		Program_Name = argv[0];
	countAndClearArgBits();
	Arg* arg_ptr;
	for (int i=1; i<argc; i++) {
		if (argv[i][0] == '-') {
			char *flag = argv[i];
			int flaglen = ::strlen(flag);
			arg_ptr = searchArgs(Args,&flag[1]);
			if (arg_ptr == NULL) {
				// if the flag is of the form -flagX, X:int, remove X and search for -flag
				// So, we perform a second search
				arg_ptr=searchArrayArgs(Args, flag);
				if(arg_ptr==NULL) {
					warning("%s Unknown switch: %s", ArgsErrStr, argv[i]);
					return (ARG_ERROR);
				}
				else if(arg_ptr == (Arg*)(-1)) return (ARG_ERROR);
				else { // found a an ARRAY type flag
					char *arg;
					if ((i+1) >= argc) arg = NULL;
					else arg = argv[++i];
					argsSwitch(arg_ptr,arg,i,Argument_Specified[arg_ptr - Args],flag);
					// xxx
					arg_ptr->incCount();
				}
			}
			else if (arg_ptr == (Arg*)(-1)) {
				warning("%s Ambiguous switch: %s",ArgsErrStr,argv[i]);
				return (ARG_ERROR);
			}
			else {
				// first check to see if it is the special parse from file
				// argument name, which is always valid.
				if (!::strncmp(ARGS_FILE_NAME,&flag[1],flaglen-1)) {
					// so argument is presumably a file name.
					char *arg=NULL;
					if ((i+1) >= argc)
						error("%s Expecting file name after %s argument flag\n",
						      ArgsErrStr,
						      ARGS_FILE_NAME);
					else
						arg = argv[++i];
					parseArgsFromFile(arg);
				} else {
					char *arg;
					if ((i+1) >= argc)
						arg = NULL;
					else
						arg = argv[++i];
					if(arg_ptr->dataStructType == ARRAY) {
						if(Error_If_No_Index_For_Array_Data_Struct_Type) {
							warning("%s Need to supply index for array type flag: %s",ArgsErrStr,flag);
							return (ARG_ERROR);
						}
						else {
							arg_ptr->arrayElmtIndex=0; // if the user specfies -i instead of -i1, we implicitly assume -i1, when the flag is of type array.
						}
					}
					argsSwitch(arg_ptr,arg,i,Argument_Specified[arg_ptr - Args],flag);
					// xxx
					arg_ptr->incCount();
				}
			}
		} else { // Go through and look for no flag case, in order.
			arg_ptr = Args;
			while (arg_ptr->flag != NULL) {
				if (arg_ptr->flag == NOFLAG) {
					// assume string type
					char *arg = argv[i];
					int dummy;
					argsSwitch(arg_ptr,arg,dummy,Argument_Specified[arg_ptr-Args],"");
					arg_ptr->flag = NOFL_FOUND;
					break;
				}
				arg_ptr++;
			}
		}
	}
	if (checkMissing())
		return ARG_MISSING;
	return ARG_OK;
}


/*-
 *-----------------------------------------------------------------------
 * parse()
 *      parses the arguments given to the program
 *
 * Preconditions:
 *      arguments must be specified
 *
 * Postconditions:
 *      arguments are parsed
 *
 * Side Effects:
 *      modifies internal object static variables.
 *
 * Results:
 *      none
 *
 *-----------------------------------------------------------------------
 */
bool Arg::parse(int argc,char** argv)
{
	ArgsRetCode rc;
	rc = parseArgsFromCommandLine(argc,argv);

	// Don't print an error message if the help flag is supplied on the cmd line
	Arg* arg_ptr=searchArgs(Args,"help");
	unsigned cnt=0;
	if(arg_ptr!=NULL) {
		cnt=arg_ptr->getCount();
	}
	if (cnt > 0) {
		// if help argument is given, print usage and exit cleanly.
		usage(NULL,false);
		exit(0);
	}

	if (cnt==0 && rc == ARG_MISSING) {
		Arg::checkMissing(true);
	}


//   ////////////////////////////////////////////////////////////////////
//   // This section deals with help arguments and info level.  Should move to a subroutine
//   ////////////////////////////////////////////////////////////////////

//   // determine the requested priority by counting the number of times
//   // the -help switch was specified on the command line
//   // As a side effect we also learn whether the -help flag was used
//   // and if so we do not print an error message along with the usage
//   // information
//   Arg* arg_ptr=searchArgs(Args,"help");
//   if(arg_ptr==NULL) {  // the -help switch is not even defined
//     if (rc == ARG_MISSING)
//       Arg::checkMissing(true);
//     // warning("WARNING: -help flag not defined");
//     // nothing to do; the required priority is set by default to the highest priority
//   }
//   else {
//     unsigned cnt=arg_ptr->getCount();
//     Requested_Priority=cnt;
//     if(Requested_Priority == 0) Requested_Priority = HIGHEST_PRIORITY; // case when no -help flag is supplied

//     if (cnt==0 && rc == ARG_MISSING)
//       Arg::checkMissing(true);
//   }

//   // Override the requested priority if an explicit flag -usageInfoLevel is supplied
//   arg_ptr=searchArgs(Args,"usageInfoLevel");
//   if(arg_ptr!=NULL) {  // the -usageInfoLevel switch is defined
//     unsigned level=*((unsigned int*)(arg_ptr->mt.ptr));
//     if (level > 0)  // because level==0 means that value wasn't set on the command line
//       Requested_Priority=level;
//   }
//   /////////////////////////////////////////////////////////////////////

	if (rc != ARG_OK) {
		//Arg::usage();
		//::exit(1);
		return false;
	}
	return true;
}


/*-
 *-----------------------------------------------------------------------
 * Test driver routine, -DMAIN to compile into program.
 *-----------------------------------------------------------------------
 */


#ifdef MAIN

/*
 * arguments without flags
 */

const char *string_fl="This is a string";
char char_fl = 'C';
float float_fl = 3.4;
double double_fl = 4.5;
bool bool_fl = false;
int int_fl = 343;

/*
 * arguments with flags
 */
const char *myString = "BARSTR";
// char *myString;
const char* strOfStr[10] = { "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten" };
float aSingle = 2.3;
double aDouble = .4;
int int1 = 3;
int int2 = 3;
bool bvalue = true;
bool abvalue[] = {false,true,false};
bool aToggle = true;
bool anotherToggle = false;
char aChar = 'c';

int input[10]={9,8,7};
int int3=3;
int int4=4;
int arg[10]={1000};

const char* testArrayOfChars="abcde";
char testArrayOfChars2[10]={'z','y','x'};


/*
 * the argument list
 */
Arg Arg::Args[] = {
	// Arguments with flags.
	Arg("arg",      Arg::Opt, arg,"An integer to test the ambiguity with -argsFile",Arg::ARRAY,0,true),
	Arg("i",      Arg::Opt, input,"An array integer",Arg::ARRAY,10,true),
	Arg("str",      Arg::Opt, strOfStr,"An array string",Arg::ARRAY,2,true),
	Arg("myString",  Arg::Opt, myString,"A string"),
	Arg("aSingle",   Arg::Opt, aSingle,"A single precision floating point num"),
	Arg("aDouble",   Arg::Opt, aDouble,"A double precision floating point num"),
	Arg("int1",      Arg::Opt, int1,"An integer"),
	Arg("int2",      Arg::Opt, int2,"A different integer"),
	Arg("in",      Arg::Opt, int3,"An integer to test disambiguation"),
	Arg("rbvalue",   Arg::Opt, bvalue,"A required boolean"),
	Arg("bvalue",    Arg::Opt, bvalue,"A boolean"),
	Arg("abvalue",    Arg::Opt, abvalue,"A boolean",Arg::ARRAY,3,true),
	Arg("aToggle",   Arg::Tog, aToggle,"a toggle"),
	Arg("anotherToggle",Arg::Tog, anotherToggle,"another toggle"),
	Arg("aChar",     Arg::Opt, aChar,"A character",Arg::SINGLE,0,false),

	// Arguments without flags. The order on the command line must be in
	// the the same as the order given here. i.e. string first, then int,
	// then bool, etc.
	Arg(Arg::Opt, char_fl,    "char"),
	Arg(Arg::Opt, float_fl,   "float"),
	Arg(Arg::Opt, double_fl,  "double"),
	Arg(Arg::Opt, bool_fl,    "bool"),
	Arg(Arg::Opt, int_fl,     "int"),

	// The argumentless argument marks the end
	// of the above list.
	Arg()
};


int main(int argc,char*argv[])
{
	Arg::parse(argc,argv);
	Arg::printArgs(Arg::Args,stdout);
	Arg::usage("a");

#if 1
	if(strOfStr != NULL)
		for (int i=0; i<2; ++i) {
			if(strOfStr[i] != NULL)
				printf("%d: %s\n",i,strOfStr[i]);
		}
#endif

#endif // #ifdef MAIN
