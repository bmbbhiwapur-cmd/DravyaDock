# --- NEW: EXPANDED HARDCODED AYURVEDIC DATABASE ---
@st.cache_data
def load_ayurvedic_db():
    hardcoded_data = [
        {
            "Master ID": "M-001", "Herb / Tree Name": "Tulsi", "Scientific Name": "Ocimum sanctum", 
            "Family": "Lamiaceae", "Phytochemical": "Eugenol", 
            "Canonical SMILES": "COC1=C(O)C=CC(CC=C)=C1", 
            "Medicinal Activity": "Antimicrobial / Antiviral", "Target Protein / Receptor Name": "Secreted Aspartyl Proteinase 1", "PDB ID": "1ZAP", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "तुलसी कटुका तिक्ता हृद्या वृष्या दाहपित्तकृत्। दीपना कुष्ठकृच्छ्रघ्न पार्श्वरुक् कफवातजित्॥", 
            "Roman Transliteration": "tulasī kaṭukā tiktā hṛdyā vṛṣyā dāhapittakṛt | dīpanā kuṣṭhakṛcchraghna pārśvaruk kaphavātajit ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu, Tikta; Virya: Usna; Vipaka: Katu", 
            "Classical Karma (Action)": "Krimighna (Antimicrobial), Jvaraghna"
        },
        {
            "Master ID": "M-002", "Herb / Tree Name": "Haldi (Turmeric)", "Scientific Name": "Curcuma longa", 
            "Family": "Zingiberaceae", "Phytochemical": "Curcumin", 
            "Canonical SMILES": "COC1=CC(=CC=C1O)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC", 
            "Medicinal Activity": "Anti-inflammatory / Anticancer", "Target Protein / Receptor Name": "Cyclooxygenase-2 (COX-2)", "PDB ID": "5KIR", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "हरिद्रा कटुका तिक्ता रूक्षोष्णा कफपित्तनुत्। वर्ण्य त्वग्दोषमेहास्रशोथपाण्डुव्रणापहा॥", 
            "Roman Transliteration": "haridrā kaṭukā tiktā rūkṣoṣṇā kaphapittanut | varṇya tvagdoṣamehāsraśothapāṇḍuvraṇāpahā ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta, Katu; Virya: Usna; Vipaka: Katu", 
            "Classical Karma (Action)": "Vishaghna, Vranaropana, Mehaghna"
        },
        {
            "Master ID": "M-003", "Herb / Tree Name": "Ashwagandha", "Scientific Name": "Withania somnifera", 
            "Family": "Solanaceae", "Phytochemical": "Withaferin A", 
            "Canonical SMILES": "CC1=C(C(=O)OC1C2C(CC3C2(CCC4C3CC(C5(C4(C=CC(=O)C5(C)O)C)O)O)C)O)C", 
            "Medicinal Activity": "Neuroprotective / Anticancer", "Target Protein / Receptor Name": "NF-kappa B Essential Modulator (NEMO)", "PDB ID": "3BRV", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "अश्वगन्धाऽनिलश्लेष्मश्वित्रशोथक्षयापहा। बल्या रसायनी तिक्ता कषायोष्णाऽतिशुक्रला॥", 
            "Roman Transliteration": "aśvagandhā'nilaśleṣmaśvitraśothakṣayāpahā | balyā rasāyanī tiktā kaṣāyoṣṇā'tiśukralā ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta, Katu, Madhura; Virya: Usna; Vipaka: Madhura", 
            "Classical Karma (Action)": "Rasayana, Balya, Vatahara"
        },
        {
            "Master ID": "M-004", "Herb / Tree Name": "Amla", "Scientific Name": "Phyllanthus emblica", 
            "Family": "Phyllanthaceae", "Phytochemical": "Gallic Acid", 
            "Canonical SMILES": "C1=C(C=C(C(=C1O)O)O)C(=O)O", 
            "Medicinal Activity": "Antioxidant / Antidiabetic", "Target Protein / Receptor Name": "Aldose Reductase", "PDB ID": "1US0", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "आमलकं कषायाम्लं मधुरं शिशिरं लघु। दाहपित्तवमीमेहशोथघ्नं रसायनम्॥", 
            "Roman Transliteration": "āmalakaṃ kaṣāyāmlaṃ madhuraṃ śiśiraṃ laghu | dāhapittavamīmehaśothaghnaṃ rasāyanam ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Amla Pradhana Pancharasa; Virya: Shita; Vipaka: Madhura", 
            "Classical Karma (Action)": "Rasayana, Pittahara, Pramehaghna"
        },
        {
            "Master ID": "M-005", "Herb / Tree Name": "Kalmegh", "Scientific Name": "Andrographis paniculata", 
            "Family": "Acanthaceae", "Phytochemical": "Andrographolide", 
            "Canonical SMILES": "CC12CCC(C(C1CCC3(C2=CC(OC3=O)C(O)CO)C)O)O", 
            "Medicinal Activity": "Hepatoprotective / Antiviral", "Target Protein / Receptor Name": "Hepatitis C Virus Protease", "PDB ID": "3M5O", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "भूनिम्बः कटुकस्तिक्तः कफपित्तज्वरापहः। रक्तदोषहरः शीतः कृमिकुष्ठविनाशनः॥", 
            "Roman Transliteration": "bhūnimbaḥ kaṭukastiktaḥ kaphapittajvarāpahaḥ | raktadoṣaharaḥ śītaḥ kṛmikuṣṭhavināśanaḥ ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta; Virya: Shita; Vipaka: Katu", 
            "Classical Karma (Action)": "Yakrituttejaka, Jvaraghna"
        },
        {
            "Master ID": "M-006", "Herb / Tree Name": "Guggul", "Scientific Name": "Commiphora mukul", 
            "Family": "Burseraceae", "Phytochemical": "Guggulsterone", 
            "Canonical SMILES": "CC12CCC3C(C1CCC2=CC(=O)C)CCC4=CC(=O)CCC34C", 
            "Medicinal Activity": "Hypolipidemic / Antidiabetic", "Target Protein / Receptor Name": "Farnesoid X Receptor (FXR)", "PDB ID": "1OSV", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गुग्गुलुः कटुकस्तिक्तो वीर्योष्णः कफवातजित्। मेदोहरः परं व्रण्यः क्लेदमेहापहो लघुः॥", 
            "Roman Transliteration": "gugguluḥ kaṭukastikto vīryoṣṇaḥ kaphavātajit | medoharaḥ paraṃ vraṇyaḥ kledamehāpaho laghuḥ ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta, Katu, Kashaya; Virya: Usna; Vipaka: Katu", 
            "Classical Karma (Action)": "Medohara, Lekhaniya"
        },
        {
            "Master ID": "M-007", "Herb / Tree Name": "Pippali", "Scientific Name": "Piper longum", 
            "Family": "Piperaceae", "Phytochemical": "Piperine", 
            "Canonical SMILES": "C1CCN(CC1)C(=O)/C=C/C=C/C2=CC3=C(C=C2)OCO3", 
            "Medicinal Activity": "Bioenhancer / Respiratory", "Target Protein / Receptor Name": "Cytochrome P450 3A4", "PDB ID": "1TQN", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "पिप्पली दीपनी वृष्या स्वादुपाका रसायनी। अनुष्णा कटुका स्निग्धा वातश्लेष्महरी लघुः॥", 
            "Roman Transliteration": "pippalī dīpanī vṛṣyā svādupākā rasāyanī | anuṣṇā kaṭukā snigdhā vātaśleṣmaharī laghuḥ ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Anushna Shita; Vipaka: Madhura", 
            "Classical Karma (Action)": "Rasayana, Kasahara, Shvasahara"
        },
        {
            "Master ID": "M-008", "Herb / Tree Name": "Sarpagandha", "Scientific Name": "Rauvolfia serpentina", 
            "Family": "Apocynaceae", "Phytochemical": "Reserpine", 
            "Canonical SMILES": "COC1C(CC2CN3CCC4=C(NC5=C4C=CC(=C5)OC)C3CC2C1C(=O)OC)OC(=O)C6=CC(=C(C(=C6)OC)OC)OC", 
            "Medicinal Activity": "Antihypertensive", "Target Protein / Receptor Name": "Vesicular Monoamine Transporter (VMAT2)", "PDB ID": "2KBI", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "सर्पगन्धा कटुस्तिक्ता रूक्षोष्णा निद्रदा परा। अपस्मारमुन्मादं च रक्तभारं च नाश्येत्॥", 
            "Roman Transliteration": "sarpagandhā kaṭustiktā rūkṣoṣṇā nidradā parā | apasmāramunmādaṃ ca raktabhāraṃ ca nāśyet ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta; Virya: Usna; Vipaka: Katu", 
            "Classical Karma (Action)": "Nidrajanana, Raktachapahara"
        },
        {
            "Master ID": "M-009", "Herb / Tree Name": "Ardraka (Ginger)", "Scientific Name": "Zingiber officinale", 
            "Family": "Zingiberaceae", "Phytochemical": "6-Gingerol", 
            "Canonical SMILES": "CCCCCC(O)CC(=O)CCC1=CC(=C(O)C=C1)OC", 
            "Medicinal Activity": "Antiemetic / Anti-inflammatory", "Target Protein / Receptor Name": "Serotonin 5-HT3 Receptor", "PDB ID": "6Y59", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "आर्द्रकं भेदनि तीक्ष्णं कटुकं रोचनं लघु। दीपनं कफवातघ्नं विबन्धनाशनं परम्॥", 
            "Roman Transliteration": "ārdrakaṃ bhedani tīkṣṇaṃ kaṭukaṃ rocanaṃ laghu | dīpanaṃ kaphavātaghnaṃ vibandhanāśanaṃ param ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Usna; Vipaka: Madhura", 
            "Classical Karma (Action)": "Deepana, Pachana, Amadoshahara"
        },
        {
            "Master ID": "M-010", "Herb / Tree Name": "Lasuna (Garlic)", "Scientific Name": "Allium sativum", 
            "Family": "Amaryllidaceae", "Phytochemical": "Allicin", 
            "Canonical SMILES": "C=CCSS(=O)CC=C", 
            "Medicinal Activity": "Cardioprotective / Antimicrobial", "Target Protein / Receptor Name": "HMG-CoA Reductase", "PDB ID": "1HW9", 
            "Sanskrit Shloka (Bhavaprakasha Nighantu)": "लशुनो बृंहणो वृष्यः स्निग्धोष्णः पाचनः सरः। रसे पाके च कटुकः तीक्ष्णो वातकफापहः॥", 
            "Roman Transliteration": "laśuno bṛṃhaṇo vṛṣyaḥ snigdhoṣṇaḥ pācanaḥ saraḥ | rase pāke ca kaṭukaḥ tīkṣṇo vātakaphāpahaḥ ||", 
            "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Amla Varjita Pancharasa; Virya: Usna; Vipaka: Katu", 
            "Classical Karma (Action)": "Hridya, Rasayana, Vatahara"
        }
    ]
    return pd.DataFrame(hardcoded_data)
