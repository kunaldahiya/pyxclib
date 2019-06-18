#include <iostream>
#include "mat.h"
#include "config.h"

using namespace std;

void reindex_VecIF( VecIF& vec, VecI& index )
{
    for( _int i=0; i<vec.size(); i++ )
        vec[i].first = index[ vec[i].first ];
    return;
}
