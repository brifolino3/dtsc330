# CamelCase
# snake_case -- this is what python programmers use
import pandas as pd

class Grants():  # class names in python are camel case (e.g. GrantReader)
    def __init__(self, path: str):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
        """
        # What is self?
        # "Self is the specific instance of the object" - Computer Scientist 
        # Store shared variables in self
        self.path = path
        self.df = self._parse(path)

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path, compression = 'zip')
        
        mapper = {
            'APPLICATION_ID': 'application_id',  # _id means an id
            'BUDGET_START': 'start_at', #  _at means a date
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',  # you will notice, homework references this
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country', 
        }
        # make column names lowercase
        # maybe combine for budget duration?
        df = df.rename(columns = mapper)[mapper.values()]
        grantees['pi_name'] = grantees['pi_name'].str.lower().str.strip()
        return df
    
    def get(self):
        """Get parsed grants"""
        return self.df   
    
    # this function will take the PI_NAMEs &
    # ensure that whenever 2 are listed, both names
    # ( ID #s ) are acknowledged
    # this will return df with one row per individual
    def getGrantees(self):
        df = self.get().copy()
        df['pi_names'] = df['pi_names'].str.split(';')
        df = df.explode('pi_names') # explode "Transform each element of a list-like to a row, replicating index values."
        df['pi_names'] = df['pi_names'].str.strip()
        df.reset_index(drop = True)
        return df
    
    # handle the missing dates
    def fillDates(self):
        df = self.get().copy()

        # make sure the format is correct
        df['start_at'] = pd.to_datetime(df['start_at'], errors = 'coerce') # missing dates .... Na

        # this will ensure that the dates will not go 
        # across more than one grant
        df = df.sort_values(['application_id', 'start_at'])

        # fill in the missing dates from the last entry
        # if this in unavailable use the following
        df['start_at'] = (
            df.groupby('application_id')['start_at']
            .ffill().bfill())
        
        return df
        



if __name__ == '__main__':
    # This is for debugging
    grants = Grants('data/RePORTER_PRJ_C_FY2025.zip')
    grantees = grants.getGrantees()
    print(grantees.head())