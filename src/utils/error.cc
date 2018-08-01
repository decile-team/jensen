/*
    $Header: /homes/bilmes/.cvsroot/gmtk_dev/miscSupport/error.cc,v 1.5 2007/08/20 02:17:11 bilmes Exp $

    Simple fatal error function.
    Jeff Bilmes <bilmes@cs.berkeley.edu>
    $Header: /homes/bilmes/.cvsroot/gmtk_dev/miscSupport/error.cc,v 1.5 2007/08/20 02:17:11 bilmes Exp $
 */


#ifndef __GNUC__
enum bool { false = 0, true = 1 };
#endif


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "error.h"

void
error(const char * const format, ...)
{
	va_list ap;
	va_start(ap,format);
	/* print out remainder of message */
	(void) vfprintf(stderr, format, ap);
	va_end(ap);
	(void) fprintf(stderr, "\n");
	(void) exit(EXIT_FAILURE);
}

void
coredump(const char * const format, ...)
{
	va_list ap;
	va_start(ap,format);
	/* print out remainder of message */
	(void) vfprintf(stderr, format, ap);
	va_end(ap);
	(void) fprintf(stderr, "\n");
	(void) abort();
}

void
warning(const char * const format, ...)
{
	va_list ap;
	va_start(ap,format);
	/* print out remainder of message */
	(void) vfprintf(stderr, format, ap);
	va_end(ap);
	(void) fprintf(stderr, "\n");
}

void
ensure(const bool condition,const char * const errorIfFail, ...)
{
	if (!condition) {
		va_list ap;
		va_start(ap,errorIfFail);
		/* print out remainder of message */
		(void) vfprintf(stderr, errorIfFail, ap);
		va_end(ap);
		(void) fprintf(stderr, "\n");
		(void) exit(EXIT_FAILURE);
	}
}


#ifdef MAIN

int main()
{
	warning("This is a warning with output %d %f (%s)\n",
	        4,4.5,"A string");
	error("This is a fatal error with output %d %f (%s), program should die after this.\n",
	      4,4.5,"A string");
	return 0;
}
#endif
