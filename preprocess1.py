import re 

def preprocess1(df):
    df = df.fillna(' ') 
    df[['add1', 'add2', 'add3']] = [' ', ' ', ' ']

    classify1 = ['시', '군', '구']
    classify2 = ['동', '면', '리', '읍']
    
    def process_address(row):
        address1 = row['address1']
        address2 = row['address2']
        if re.search(r'[^0-9-]', address2):
            address1 += ' ' + address2[0] 
            address2 = ' '.join(address2.split()[1:])
        
        address1_list = address1.split()
        row['add1'] = address1_list.pop(0)     
        
        addr2 = " ".join([addr for addr in address1_list if addr[-1] in classify1])
        if addr2: 
            row['add2'] = addr2.strip() 

        for address in address1_list: 
            if address[-1] in classify2 and row['add3'] == ' ':
                row['add3'] = address
                
        return row

    df = df.apply(process_address, axis=1)

    df['manage_cost_inc'] = df['manage_cost_inc'].str.replace(',', ', ', regex=True) 
    df['options'] = df['options'].str.replace(',', ', ', regex=True) 
    df['description'] = df['description'].str.split().str.join(' ') 
    df['title'] = df['title'].str.split().str.join(' ') 

    return df
