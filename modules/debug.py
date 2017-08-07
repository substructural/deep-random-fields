import inspect


#---------------------------------------------------------------------------------------------------


def print_variables( local_variables ):

    indent = "   "
    width = max( [ len( name ) for name in local_variables ] )
    entry = indent + "{0:" + str( width ) + "} : {1}"
    for name in sorted( local_variables ):
        value = local_variables[ name ]
        representation = str( value ).split( '\n' )
        if len( representation ) == 1:
            print( entry.format( name, representation[ 0 ] ) )
        else:
            print( entry.format( name, "" ) )
            print( indent + "{" )
            for line in representation:
                print( indent + indent + line )
            print( indent + "}" )


def locals( display = False ):

    caller = inspect.currentframe().f_back
    if caller:
        variables = caller.f_locals
        if display:
            print( "\n>>> " + caller.f_code.co_name + ":\n" )
            print_variables( variables )
            print( "" )
        return variables
    else:
        return {}
                             


#---------------------------------------------------------------------------------------------------
