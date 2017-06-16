#===================================================================================================
# simple logging out put module



import sys


#---------------------------------------------------------------------------------------------------

class Log( object ):


    def __init__( self, file_object = None, in_memory = False ):

        self.__file_object = file_object
        self.__store = [] if in_memory else None


    @property
    def entries( self ):

        if not self.__store is None:
            return "".join( self.__store ) 
        else:
            return ""
    

    def write( self, text ):

        if self.__store is not None:
            self.__store.append( text )

        if self.__file_object:
            self.__file_object.write( text )


    @staticmethod
    def underline( text, character ):

        return text + '\n' + ''.join( [ character for c in text ] )
        

    def section( self, section_header ):

        self.write( "\n\n\n\n" + Log.underline( section_header, '=' ) + "\n\n" )


    def subsection( self, subsection_header ):

        self.write( "\n\n" + Log.underline( subsection_header, '-' ) + "\n\n" )


    def entry( self, entry ):

        self.write( "  " + entry + "\n" )


    def item( self, item ):

        self.write( "  - " + item + "\n" )


    def items( self, items ):

        for item in items:
            self.item( item )


    def record( self, record ):

        entries = record if isinstance( record, dict ) else record.__dict__ 
        width = max( [ len( key ) for key in entries ] )
        line = "    {0:" + str( width ) + "} : {1}\n"
        for entry in entries:
            self.write( line.format( entry, entries[ entry ] ) )


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    class Structure:

        def __init__( self, alpha, beta, gamma, delta, epsilon ):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
            self.epsilon = epsilon

    fields = { 'alpha':1, 'beta': 2, 'gamma': 3, 'delta': 4, 'epsilon': 5 }
    structure = Structure( **fields )

    log = Log( sys.stdout )
    
    log.section( "section" )

    log.subsection( "subsection 1" )
    log.items( [ "item 1", "item 2" ] )

    log.subsection( "subsection 2" )
    log.record( fields )

    log.subsection( "subsection 3" )
    log.record( structure )
