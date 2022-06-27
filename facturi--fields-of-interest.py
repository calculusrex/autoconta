

câmpuri_de_interes__facturi_de_marfă = {
    'header': ['furnizor',
               'data_emitere',
               'nr_factură'],
    'body': ['denumire_produs',
             'cantitatea',
             'valoare_fără_tva_(preț_cumpărare)',
             'preț_vânzare']
}

câmpuri_de_interes__facturi_de_servicii = {
    'header': ['furnizor',
               'data_emitere',
               'nr_factură'],
    'body': ['denumire_serviciu',
             'cantitatea',
             'valoare_fără_tva']
}

câmpuri_de_interes__facturi_de_piese_auto = {
    'header': ['furnizor',
               'data_emitere',
               'nr_factură'],
    'body': ['denumire_piesă',
             'cantitatea',
             'valoare_fără_tva']
}

câmpuri_de_interes__facturi_de_consumabile = {
    'header': ['furnizor',
               'data_emitere',
               'nr_factură'],
    'body': ['denumire_piesă',
             'cantitatea',
             'valoare_fără_tva']
}


proceduri_verificare = [
    'total',
    'adaos_comercial_pozitiv?'
]

conturi_segregare_articole = {
    'marfă': {
        '19%tva': 371.1, # alcool, tutun, tot ce nu e mâncare
        '9%tva': 371.2, # anumite articole mâncare
        '5%tva': 371.3, # anumite articole mâncare
    },
    'consumabile': 3028,
    'piese_de_schimb': 3024,
    'servicii': 628,
    'obiecte_de_inventar': 303,
    'cheltuieli_de_protocol': 6231,
    'combustibil': 3022,
}
