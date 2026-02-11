"""
Configuration module for Flight Price Tracker.

Loads environment variables and defines application constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API CREDENTIALS
# =============================================================================
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
AMADEUS_BASE_URL = "https://api.amadeus.com"

# Note: Credentials validated at runtime in data_fetcher.py, not at import time
# This allows the app to start and show helpful error messages

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_PATH = os.getenv("DATABASE_PATH", "flight_tracker.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# =============================================================================
# API RATE LIMITING
# =============================================================================
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "5"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.3"))  # seconds between API calls
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# =============================================================================
# FLIGHT SEARCH PARAMETERS
# =============================================================================
MAX_SEARCH_DAYS = 364  # Rolling window for price tracking
DEFAULT_TRIP_DURATIONS = [3, 5, 7, 10, 14]
DEFAULT_CABIN_CLASS = "ECONOMY"
CURRENCY = "USD"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================
LEAD_TIME_BUCKETS = [
    (0, 7, "0-7 days"),
    (8, 14, "8-14 days"),
    (15, 21, "15-21 days"),
    (22, 30, "22-30 days"),
    (31, 45, "31-45 days"),
    (46, 60, "46-60 days"),
    (61, 90, "61-90 days"),
    (91, 120, "91-120 days"),
    (121, 180, "121-180 days"),
    (181, 365, "181+ days"),
]

DEAL_THRESHOLD_PERCENT = 15  # Flag as deal if >15% below average
MIN_DATA_POINTS_FOR_ANALYSIS = 10
CACHE_TTL_SECONDS = 3600  # 1 hour cache for analysis results

# =============================================================================
# VALID IATA AIRPORT CODES (300+ airports worldwide)
# =============================================================================
VALID_IATA_CODES = {
    # =========================================================================
    # UNITED STATES
    # =========================================================================
    "JFK": "New York John F. Kennedy",
    "LGA": "New York LaGuardia",
    "EWR": "Newark Liberty",
    "LAX": "Los Angeles International",
    "SFO": "San Francisco International",
    "OAK": "Oakland International",
    "SJC": "San Jose Mineta",
    "ORD": "Chicago O'Hare",
    "MDW": "Chicago Midway",
    "DFW": "Dallas/Fort Worth",
    "DAL": "Dallas Love Field",
    "IAH": "Houston George Bush",
    "HOU": "Houston Hobby",
    "DEN": "Denver International",
    "SEA": "Seattle-Tacoma",
    "MIA": "Miami International",
    "FLL": "Fort Lauderdale",
    "PBI": "West Palm Beach",
    "ATL": "Atlanta Hartsfield-Jackson",
    "BOS": "Boston Logan",
    "IAD": "Washington Dulles",
    "DCA": "Washington Reagan",
    "BWI": "Baltimore-Washington",
    "PHX": "Phoenix Sky Harbor",
    "LAS": "Las Vegas Harry Reid",
    "MCO": "Orlando International",
    "SFB": "Orlando Sanford",
    "MSP": "Minneapolis-Saint Paul",
    "DTW": "Detroit Metropolitan",
    "PHL": "Philadelphia International",
    "CLT": "Charlotte Douglas",
    "SAN": "San Diego International",
    "TPA": "Tampa International",
    "PDX": "Portland International",
    "HNL": "Honolulu International",
    "OGG": "Maui Kahului",
    "LIH": "Lihue Kauai",
    "KOA": "Kona International",
    "SLC": "Salt Lake City",
    "AUS": "Austin-Bergstrom",
    "SAT": "San Antonio International",
    "RDU": "Raleigh-Durham",
    "BNA": "Nashville International",
    "MCI": "Kansas City International",
    "STL": "St. Louis Lambert",
    "IND": "Indianapolis International",
    "CMH": "Columbus John Glenn",
    "CLE": "Cleveland Hopkins",
    "PIT": "Pittsburgh International",
    "CVG": "Cincinnati/Northern Kentucky",
    "MKE": "Milwaukee Mitchell",
    "SNA": "Orange County John Wayne",
    "BUR": "Burbank Hollywood",
    "ONT": "Ontario International",
    "SMF": "Sacramento International",
    "RNO": "Reno-Tahoe",
    "ABQ": "Albuquerque Sunport",
    "ELP": "El Paso International",
    "TUS": "Tucson International",
    "OKC": "Oklahoma City Will Rogers",
    "TUL": "Tulsa International",
    "ORF": "Norfolk International",
    "RIC": "Richmond International",
    "JAX": "Jacksonville International",
    "RSW": "Fort Myers Southwest Florida",
    "MSY": "New Orleans Louis Armstrong",
    "MEM": "Memphis International",
    "BHM": "Birmingham-Shuttlesworth",
    "PVD": "Providence T.F. Green",
    "BDL": "Hartford Bradley",
    "SYR": "Syracuse Hancock",
    "BUF": "Buffalo Niagara",
    "ROC": "Rochester Greater",
    "ALB": "Albany International",
    "PWM": "Portland Maine Jetport",
    "MHT": "Manchester-Boston Regional",
    "BTV": "Burlington International",
    "ANC": "Anchorage Ted Stevens",
    "FAI": "Fairbanks International",
    # =========================================================================
    # CANADA
    # =========================================================================
    "YYZ": "Toronto Pearson",
    "YYC": "Calgary International",
    "YVR": "Vancouver International",
    "YUL": "Montreal Trudeau",
    "YOW": "Ottawa Macdonald-Cartier",
    "YEG": "Edmonton International",
    "YWG": "Winnipeg Richardson",
    "YHZ": "Halifax Stanfield",
    "YQB": "Quebec City Jean Lesage",
    "YYJ": "Victoria International",
    "YXE": "Saskatoon John G. Diefenbaker",
    "YQR": "Regina International",
    "YLW": "Kelowna International",
    "YXU": "London Ontario",
    "YKF": "Waterloo Region",
    # =========================================================================
    # MEXICO
    # =========================================================================
    "MEX": "Mexico City International",
    "CUN": "Cancun International",
    "GDL": "Guadalajara Miguel Hidalgo",
    "MTY": "Monterrey International",
    "TIJ": "Tijuana International",
    "SJD": "Los Cabos International",
    "PVR": "Puerto Vallarta",
    "MID": "Merida International",
    "CZM": "Cozumel International",
    "ACA": "Acapulco International",
    "ZIH": "Zihuatanejo International",
    "OAX": "Oaxaca Xoxocotlan",
    "BJX": "Leon Guanajuato",
    "QRO": "Queretaro International",
    "AGU": "Aguascalientes International",
    "SLP": "San Luis Potosi",
    "HMO": "Hermosillo International",
    "CUL": "Culiacan International",
    "MZT": "Mazatlan International",
    "VER": "Veracruz International",
    "VSA": "Villahermosa International",
    "TAP": "Tapachula International",
    # =========================================================================
    # CARIBBEAN
    # =========================================================================
    "SJU": "San Juan Luis Munoz Marin",
    "STT": "St. Thomas Cyril E. King",
    "STX": "St. Croix Henry E. Rohlsen",
    "NAS": "Nassau Lynden Pindling",
    "MBJ": "Montego Bay Sangster",
    "KIN": "Kingston Norman Manley",
    "PUJ": "Punta Cana International",
    "SDQ": "Santo Domingo Las Americas",
    "HAV": "Havana Jose Marti",
    "VRA": "Varadero Juan Gualberto Gomez",
    "AUA": "Aruba Queen Beatrix",
    "CUR": "Curacao Hato",
    "SXM": "St. Maarten Princess Juliana",
    "POS": "Port of Spain Piarco",
    "BGI": "Barbados Grantley Adams",
    "GCM": "Grand Cayman Owen Roberts",
    "BDA": "Bermuda L.F. Wade",
    "FPO": "Freeport Grand Bahama",
    "ELH": "North Eleuthera",
    "GGT": "George Town Exuma",
    "SAL": "San Salvador Oscar Romero",
    "GUA": "Guatemala City La Aurora",
    "SAP": "San Pedro Sula Ramon Villeda",
    "TGU": "Tegucigalpa Toncontin",
    "MGA": "Managua Augusto Sandino",
    "SJO": "San Jose Costa Rica Juan Santamaria",
    "LIR": "Liberia Costa Rica",
    "PTY": "Panama City Tocumen",
    "BZE": "Belize City Philip Goldson",
    # =========================================================================
    # SOUTH AMERICA
    # =========================================================================
    "GRU": "Sao Paulo Guarulhos",
    "CGH": "Sao Paulo Congonhas",
    "VCP": "Campinas Viracopos",
    "GIG": "Rio de Janeiro Galeao",
    "SDU": "Rio de Janeiro Santos Dumont",
    "BSB": "Brasilia International",
    "CNF": "Belo Horizonte Confins",
    "SSA": "Salvador International",
    "REC": "Recife Guararapes",
    "FOR": "Fortaleza Pinto Martins",
    "POA": "Porto Alegre Salgado Filho",
    "CWB": "Curitiba Afonso Pena",
    "FLN": "Florianopolis Hercilio Luz",
    "MAO": "Manaus Eduardo Gomes",
    "BEL": "Belem Val de Cans",
    "NAT": "Natal Augusto Severo",
    "EZE": "Buenos Aires Ezeiza",
    "AEP": "Buenos Aires Aeroparque",
    "COR": "Cordoba Ingeniero Ambrosio Taravella",
    "MDZ": "Mendoza El Plumerillo",
    "BRC": "Bariloche Teniente Luis Candelaria",
    "IGR": "Puerto Iguazu",
    "USH": "Ushuaia Malvinas Argentinas",
    "SCL": "Santiago Arturo Merino Benitez",
    "LIM": "Lima Jorge Chavez",
    "CUZ": "Cusco Alejandro Velasco Astete",
    "BOG": "Bogota El Dorado",
    "MDE": "Medellin Jose Maria Cordova",
    "CTG": "Cartagena Rafael Nunez",
    "CLO": "Cali Alfonso Bonilla Aragon",
    "UIO": "Quito Mariscal Sucre",
    "GYE": "Guayaquil Jose Joaquin de Olmedo",
    "CCS": "Caracas Simon Bolivar",
    "LPB": "La Paz El Alto",
    "VVI": "Santa Cruz Viru Viru",
    "ASU": "Asuncion Silvio Pettirossi",
    "MVD": "Montevideo Carrasco",
    # =========================================================================
    # EUROPE - UNITED KINGDOM & IRELAND
    # =========================================================================
    "LHR": "London Heathrow",
    "LGW": "London Gatwick",
    "STN": "London Stansted",
    "LTN": "London Luton",
    "LCY": "London City",
    "SEN": "London Southend",
    "MAN": "Manchester",
    "BHX": "Birmingham",
    "EDI": "Edinburgh",
    "GLA": "Glasgow",
    "BRS": "Bristol",
    "LPL": "Liverpool John Lennon",
    "NCL": "Newcastle",
    "LBA": "Leeds Bradford",
    "EMA": "East Midlands",
    "BFS": "Belfast International",
    "BHD": "Belfast City George Best",
    "ABZ": "Aberdeen",
    "INV": "Inverness",
    "CWL": "Cardiff",
    "DUB": "Dublin",
    "ORK": "Cork",
    "SNN": "Shannon",
    "KNO": "Knock Ireland West",
    # =========================================================================
    # EUROPE - FRANCE
    # =========================================================================
    "CDG": "Paris Charles de Gaulle",
    "ORY": "Paris Orly",
    "BVA": "Paris Beauvais",
    "NCE": "Nice Cote d'Azur",
    "LYS": "Lyon Saint-Exupery",
    "MRS": "Marseille Provence",
    "TLS": "Toulouse Blagnac",
    "BOD": "Bordeaux Merignac",
    "NTE": "Nantes Atlantique",
    "SXB": "Strasbourg",
    "LIL": "Lille",
    "MPL": "Montpellier",
    "BIQ": "Biarritz",
    "AJA": "Ajaccio Napoleon Bonaparte",
    "BIA": "Bastia Poretta",
    # =========================================================================
    # EUROPE - GERMANY
    # =========================================================================
    "FRA": "Frankfurt International",
    "MUC": "Munich Franz Josef Strauss",
    "TXL": "Berlin Tegel (closed)",
    "BER": "Berlin Brandenburg",
    "DUS": "Dusseldorf",
    "HAM": "Hamburg",
    "CGN": "Cologne Bonn",
    "STR": "Stuttgart",
    "HAJ": "Hannover",
    "NUE": "Nuremberg",
    "LEJ": "Leipzig/Halle",
    "DRS": "Dresden",
    "BRE": "Bremen",
    "DTM": "Dortmund",
    "FMO": "Munster Osnabruck",
    # =========================================================================
    # EUROPE - SPAIN & PORTUGAL
    # =========================================================================
    "MAD": "Madrid Barajas",
    "BCN": "Barcelona El Prat",
    "PMI": "Palma de Mallorca",
    "AGP": "Malaga Costa del Sol",
    "ALC": "Alicante Elche",
    "VLC": "Valencia",
    "SVQ": "Seville San Pablo",
    "BIO": "Bilbao",
    "IBZ": "Ibiza",
    "TFS": "Tenerife South",
    "TFN": "Tenerife North",
    "LPA": "Gran Canaria",
    "ACE": "Lanzarote",
    "FUE": "Fuerteventura",
    "SCQ": "Santiago de Compostela",
    "LIS": "Lisbon Humberto Delgado",
    "OPO": "Porto Francisco Sa Carneiro",
    "FAO": "Faro",
    "FNC": "Funchal Madeira",
    "PDL": "Ponta Delgada Azores",
    # =========================================================================
    # EUROPE - ITALY
    # =========================================================================
    "FCO": "Rome Fiumicino",
    "CIA": "Rome Ciampino",
    "MXP": "Milan Malpensa",
    "LIN": "Milan Linate",
    "BGY": "Milan Bergamo Orio al Serio",
    "VCE": "Venice Marco Polo",
    "TSF": "Venice Treviso",
    "NAP": "Naples Capodichino",
    "BLQ": "Bologna Guglielmo Marconi",
    "FLR": "Florence Peretola",
    "PSA": "Pisa Galileo Galilei",
    "TRN": "Turin Caselle",
    "CTA": "Catania Fontanarossa",
    "PMO": "Palermo Falcone Borsellino",
    "CAG": "Cagliari Elmas",
    "OLB": "Olbia Costa Smeralda",
    "BRI": "Bari Karol Wojtyla",
    "VRN": "Verona Villafranca",
    "GOA": "Genoa Cristoforo Colombo",
    # =========================================================================
    # EUROPE - NETHERLANDS, BELGIUM, LUXEMBOURG
    # =========================================================================
    "AMS": "Amsterdam Schiphol",
    "RTM": "Rotterdam The Hague",
    "EIN": "Eindhoven",
    "BRU": "Brussels",
    "CRL": "Brussels South Charleroi",
    "ANR": "Antwerp",
    "LUX": "Luxembourg Findel",
    # =========================================================================
    # EUROPE - SWITZERLAND & AUSTRIA
    # =========================================================================
    "ZRH": "Zurich",
    "GVA": "Geneva",
    "BSL": "Basel-Mulhouse-Freiburg",
    "BRN": "Bern",
    "VIE": "Vienna Schwechat",
    "SZG": "Salzburg W.A. Mozart",
    "INN": "Innsbruck",
    "GRZ": "Graz",
    "LNZ": "Linz",
    # =========================================================================
    # EUROPE - SCANDINAVIA
    # =========================================================================
    "CPH": "Copenhagen Kastrup",
    "BLL": "Billund",
    "AAL": "Aalborg",
    "OSL": "Oslo Gardermoen",
    "TRD": "Trondheim Vaernes",
    "BGO": "Bergen Flesland",
    "SVG": "Stavanger Sola",
    "TOS": "Tromso",
    "ARN": "Stockholm Arlanda",
    "BMA": "Stockholm Bromma",
    "GOT": "Gothenburg Landvetter",
    "MMX": "Malmo Sturup",
    "HEL": "Helsinki Vantaa",
    "TMP": "Tampere Pirkkala",
    "TKU": "Turku",
    "OUL": "Oulu",
    "RVN": "Rovaniemi",
    "KEF": "Reykjavik Keflavik",
    # =========================================================================
    # EUROPE - EASTERN EUROPE
    # =========================================================================
    "WAW": "Warsaw Chopin",
    "WMI": "Warsaw Modlin",
    "KRK": "Krakow John Paul II",
    "GDN": "Gdansk Lech Walesa",
    "WRO": "Wroclaw",
    "POZ": "Poznan",
    "KTW": "Katowice",
    "PRG": "Prague Vaclav Havel",
    "BRQ": "Brno",
    "BUD": "Budapest Ferenc Liszt",
    "OTP": "Bucharest Otopeni",
    "CLJ": "Cluj-Napoca",
    "SOF": "Sofia",
    "VAR": "Varna",
    "ZAG": "Zagreb Franjo Tudman",
    "SPU": "Split",
    "DBV": "Dubrovnik",
    "PUY": "Pula",
    "ZAD": "Zadar",
    "BEG": "Belgrade Nikola Tesla",
    "LJU": "Ljubljana Joze Pucnik",
    "SKP": "Skopje International",
    "TIA": "Tirana Mother Teresa",
    "SJJ": "Sarajevo",
    "KIV": "Chisinau",
    "RIX": "Riga International",
    "VNO": "Vilnius",
    "TLL": "Tallinn Lennart Meri",
    "KBP": "Kyiv Boryspil",
    "IEV": "Kyiv Zhuliany",
    "ODS": "Odesa",
    "LWO": "Lviv Danylo Halytskyi",
    # =========================================================================
    # EUROPE - GREECE & CYPRUS
    # =========================================================================
    "ATH": "Athens Eleftherios Venizelos",
    "SKG": "Thessaloniki Macedonia",
    "HER": "Heraklion Nikos Kazantzakis",
    "CHQ": "Chania Ioannis Daskalogiannis",
    "RHO": "Rhodes Diagoras",
    "CFU": "Corfu Ioannis Kapodistrias",
    "ZTH": "Zakynthos Dionysios Solomos",
    "KGS": "Kos Hippocrates",
    "JMK": "Mykonos",
    "JTR": "Santorini",
    "LCA": "Larnaca",
    "PFO": "Paphos",
    # =========================================================================
    # EUROPE - TURKEY
    # =========================================================================
    "IST": "Istanbul Airport",
    "SAW": "Istanbul Sabiha Gokcen",
    "ESB": "Ankara Esenboga",
    "AYT": "Antalya",
    "DLM": "Dalaman",
    "BJV": "Bodrum Milas",
    "ADB": "Izmir Adnan Menderes",
    "TZX": "Trabzon",
    "GZT": "Gaziantep",
    "ADA": "Adana Sakirpasa",
    # =========================================================================
    # MIDDLE EAST
    # =========================================================================
    "DXB": "Dubai International",
    "DWC": "Dubai Al Maktoum",
    "AUH": "Abu Dhabi International",
    "SHJ": "Sharjah International",
    "DOH": "Doha Hamad",
    "BAH": "Bahrain International",
    "KWI": "Kuwait International",
    "MCT": "Muscat International",
    "SLL": "Salalah",
    "RUH": "Riyadh King Khalid",
    "JED": "Jeddah King Abdulaziz",
    "DMM": "Dammam King Fahd",
    "MED": "Medina Prince Mohammad",
    "AMM": "Amman Queen Alia",
    "BEY": "Beirut Rafic Hariri",
    "TLV": "Tel Aviv Ben Gurion",
    "BGW": "Baghdad International",
    "EBL": "Erbil International",
    "BSR": "Basra International",
    "THR": "Tehran Imam Khomeini",
    "IKA": "Tehran Imam Khomeini",
    "MHD": "Mashhad International",
    "SYZ": "Shiraz International",
    "ISF": "Isfahan International",
    # =========================================================================
    # AFRICA - NORTH
    # =========================================================================
    "CAI": "Cairo International",
    "HRG": "Hurghada International",
    "SSH": "Sharm el-Sheikh International",
    "LXR": "Luxor International",
    "ALG": "Algiers Houari Boumediene",
    "ORN": "Oran Ahmed Ben Bella",
    "TUN": "Tunis Carthage",
    "CMN": "Casablanca Mohammed V",
    "RAK": "Marrakech Menara",
    "AGA": "Agadir Al Massira",
    "TNG": "Tangier Ibn Battouta",
    "FEZ": "Fes Saiss",
    "TIP": "Tripoli International",
    # =========================================================================
    # AFRICA - WEST & CENTRAL
    # =========================================================================
    "LOS": "Lagos Murtala Muhammed",
    "ABV": "Abuja Nnamdi Azikiwe",
    "ACC": "Accra Kotoka",
    "ABJ": "Abidjan Felix Houphouet-Boigny",
    "DKR": "Dakar Blaise Diagne",
    "COO": "Cotonou Cadjehoun",
    "LFW": "Lome Gnassingbe Eyadema",
    "OUA": "Ouagadougou",
    "BKO": "Bamako Modibo Keita",
    "NKC": "Nouakchott Oumtounsy",
    "FNA": "Freetown Lungi",
    "ROB": "Monrovia Roberts",
    "CKY": "Conakry Gbessia",
    "BSG": "Bata",
    "LBV": "Libreville Leon M'ba",
    "DLA": "Douala International",
    "NSI": "Yaounde Nsimalen",
    "FIH": "Kinshasa N'djili",
    "BZV": "Brazzaville Maya-Maya",
    "LUN": "Lusaka Kenneth Kaunda",
    "HRE": "Harare Robert Gabriel Mugabe",
    # =========================================================================
    # AFRICA - EAST
    # =========================================================================
    "ADD": "Addis Ababa Bole",
    "NBO": "Nairobi Jomo Kenyatta",
    "MBA": "Mombasa Moi",
    "DAR": "Dar es Salaam Julius Nyerere",
    "ZNZ": "Zanzibar Abeid Amani Karume",
    "JRO": "Kilimanjaro International",
    "EBB": "Entebbe International",
    "KGL": "Kigali International",
    "BJM": "Bujumbura International",
    "SEZ": "Mahe Seychelles",
    "MRU": "Mauritius Sir Seewoosagur Ramgoolam",
    "RUN": "Reunion Roland Garros",
    "TNR": "Antananarivo Ivato",
    "NOS": "Nosy Be Fascene",
    # =========================================================================
    # AFRICA - SOUTH
    # =========================================================================
    "JNB": "Johannesburg O.R. Tambo",
    "CPT": "Cape Town International",
    "DUR": "Durban King Shaka",
    "PLZ": "Port Elizabeth",
    "GRJ": "George",
    "WDH": "Windhoek Hosea Kutako",
    "GBE": "Gaborone Sir Seretse Khama",
    "MTS": "Manzini King Mswati III",
    "MPM": "Maputo International",
    "LAD": "Luanda Quatro de Fevereiro",
    # =========================================================================
    # ASIA - EAST ASIA
    # =========================================================================
    "PEK": "Beijing Capital",
    "PKX": "Beijing Daxing",
    "PVG": "Shanghai Pudong",
    "SHA": "Shanghai Hongqiao",
    "CAN": "Guangzhou Baiyun",
    "SZX": "Shenzhen Bao'an",
    "CTU": "Chengdu Shuangliu",
    "TFU": "Chengdu Tianfu",
    "CKG": "Chongqing Jiangbei",
    "XIY": "Xi'an Xianyang",
    "HGH": "Hangzhou Xiaoshan",
    "NKG": "Nanjing Lukou",
    "WUH": "Wuhan Tianhe",
    "CSX": "Changsha Huanghua",
    "KMG": "Kunming Changshui",
    "XMN": "Xiamen Gaoqi",
    "SYX": "Sanya Phoenix",
    "HAK": "Haikou Meilan",
    "TSN": "Tianjin Binhai",
    "SHE": "Shenyang Taoxian",
    "DLC": "Dalian Zhoushuizi",
    "TAO": "Qingdao Jiaodong",
    "HRB": "Harbin Taiping",
    "CGO": "Zhengzhou Xinzheng",
    "NNG": "Nanning Wuxu",
    "FOC": "Fuzhou Changle",
    "HET": "Hohhot Baita",
    "URC": "Urumqi Diwopu",
    "LHW": "Lanzhou Zhongchuan",
    "HKG": "Hong Kong International",
    "MFM": "Macau International",
    "TPE": "Taipei Taoyuan",
    "TSA": "Taipei Songshan",
    "KHH": "Kaohsiung International",
    "RMQ": "Taichung International",
    "NRT": "Tokyo Narita",
    "HND": "Tokyo Haneda",
    "KIX": "Osaka Kansai",
    "ITM": "Osaka Itami",
    "NGO": "Nagoya Chubu Centrair",
    "FUK": "Fukuoka",
    "CTS": "Sapporo New Chitose",
    "OKA": "Okinawa Naha",
    "HIJ": "Hiroshima",
    "KOJ": "Kagoshima",
    "SDJ": "Sendai",
    "KMJ": "Kumamoto",
    "ICN": "Seoul Incheon",
    "GMP": "Seoul Gimpo",
    "PUS": "Busan Gimhae",
    "CJU": "Jeju International",
    "TAE": "Daegu International",
    "ULN": "Ulaanbaatar Chinggis Khaan",
    # =========================================================================
    # ASIA - SOUTHEAST
    # =========================================================================
    "SIN": "Singapore Changi",
    "KUL": "Kuala Lumpur International",
    "SZB": "Kuala Lumpur Sultan Abdul Aziz Shah",
    "PEN": "Penang International",
    "LGK": "Langkawi International",
    "BKI": "Kota Kinabalu International",
    "KCH": "Kuching International",
    "JHB": "Johor Bahru Senai",
    "BKK": "Bangkok Suvarnabhumi",
    "DMK": "Bangkok Don Mueang",
    "CNX": "Chiang Mai International",
    "HKT": "Phuket International",
    "USM": "Koh Samui",
    "KBV": "Krabi International",
    "HDY": "Hat Yai International",
    "CEI": "Chiang Rai Mae Fah Luang",
    "SGN": "Ho Chi Minh City Tan Son Nhat",
    "HAN": "Hanoi Noi Bai",
    "DAD": "Da Nang International",
    "CXR": "Nha Trang Cam Ranh",
    "PQC": "Phu Quoc International",
    "MNL": "Manila Ninoy Aquino",
    "CEB": "Cebu Mactan",
    "DVO": "Davao Francisco Bangoy",
    "ILO": "Iloilo International",
    "KLO": "Kalibo International",
    "CGK": "Jakarta Soekarno-Hatta",
    "HLP": "Jakarta Halim Perdanakusuma",
    "DPS": "Bali Ngurah Rai",
    "SUB": "Surabaya Juanda",
    "JOG": "Yogyakarta Adisucipto",
    "YIA": "Yogyakarta International",
    "UPG": "Makassar Sultan Hasanuddin",
    "BPN": "Balikpapan Sultan Aji Muhammad Sulaiman",
    "MDC": "Manado Sam Ratulangi",
    "LOP": "Lombok International",
    "PDG": "Padang Minangkabau",
    "PLM": "Palembang Sultan Mahmud Badaruddin II",
    "PKU": "Pekanbaru Sultan Syarif Kasim II",
    "BTJ": "Banda Aceh Sultan Iskandar Muda",
    "REP": "Siem Reap Angkor",
    "PNH": "Phnom Penh International",
    "RGN": "Yangon International",
    "MDL": "Mandalay International",
    "VTE": "Vientiane Wattay",
    "LPQ": "Luang Prabang International",
    "BWN": "Bandar Seri Begawan",
    "DIL": "Dili Presidente Nicolau Lobato",
    # =========================================================================
    # ASIA - SOUTH
    # =========================================================================
    "DEL": "Delhi Indira Gandhi",
    "BOM": "Mumbai Chhatrapati Shivaji",
    "BLR": "Bangalore Kempegowda",
    "MAA": "Chennai International",
    "CCU": "Kolkata Netaji Subhas Chandra Bose",
    "HYD": "Hyderabad Rajiv Gandhi",
    "AMD": "Ahmedabad Sardar Vallabhbhai Patel",
    "PNQ": "Pune International",
    "GOI": "Goa Dabolim",
    "COK": "Kochi International",
    "TRV": "Thiruvananthapuram International",
    "JAI": "Jaipur International",
    "LKO": "Lucknow Chaudhary Charan Singh",
    "IXC": "Chandigarh International",
    "SXR": "Srinagar Sheikh ul-Alam",
    "ATQ": "Amritsar Sri Guru Ram Dass Jee",
    "GAU": "Guwahati Lokpriya Gopinath Bordoloi",
    "IXB": "Bagdogra",
    "BBI": "Bhubaneswar Biju Patnaik",
    "PAT": "Patna Jay Prakash Narayan",
    "VNS": "Varanasi Lal Bahadur Shastri",
    "IXE": "Mangalore International",
    "CMB": "Colombo Bandaranaike",
    "MLE": "Male Velana",
    "DAC": "Dhaka Hazrat Shahjalal",
    "CGP": "Chittagong Shah Amanat",
    "KTM": "Kathmandu Tribhuvan",
    "ISB": "Islamabad International",
    "KHI": "Karachi Jinnah",
    "LHE": "Lahore Allama Iqbal",
    "KBL": "Kabul Hamid Karzai",
    # =========================================================================
    # ASIA - CENTRAL
    # =========================================================================
    "TAS": "Tashkent Islam Karimov",
    "ALA": "Almaty International",
    "NQZ": "Astana Nursultan Nazarbayev",
    "FRU": "Bishkek Manas",
    "DYU": "Dushanbe International",
    "ASB": "Ashgabat International",
    "GYD": "Baku Heydar Aliyev",
    "TBS": "Tbilisi International",
    "EVN": "Yerevan Zvartnots",
    # =========================================================================
    # OCEANIA - AUSTRALIA
    # =========================================================================
    "SYD": "Sydney Kingsford Smith",
    "MEL": "Melbourne Tullamarine",
    "AVV": "Avalon Melbourne",
    "BNE": "Brisbane International",
    "PER": "Perth International",
    "ADL": "Adelaide International",
    "OOL": "Gold Coast Coolangatta",
    "CNS": "Cairns International",
    "CBR": "Canberra International",
    "HBA": "Hobart International",
    "DRW": "Darwin International",
    "TSV": "Townsville",
    "NTL": "Newcastle Williamtown",
    "ASP": "Alice Springs",
    "AYQ": "Ayers Rock Connellan",
    "MCY": "Sunshine Coast",
    "LST": "Launceston",
    # =========================================================================
    # OCEANIA - NEW ZEALAND & PACIFIC
    # =========================================================================
    "AKL": "Auckland International",
    "WLG": "Wellington International",
    "CHC": "Christchurch International",
    "ZQN": "Queenstown",
    "DUD": "Dunedin International",
    "ROT": "Rotorua International",
    "NPE": "Napier Hawke's Bay",
    "NPL": "New Plymouth",
    "NSN": "Nelson",
    "PMR": "Palmerston North",
    "HLZ": "Hamilton International",
    "TRG": "Tauranga",
    "NAN": "Nadi International Fiji",
    "SUV": "Suva Nausori Fiji",
    "PPT": "Papeete Faa'a Tahiti",
    "NOU": "Noumea La Tontouta",
    "APW": "Apia Faleolo Samoa",
    "TBU": "Tongatapu Fua'amotu Tonga",
    "VLI": "Port Vila Bauerfield Vanuatu",
    "HIR": "Honiara Henderson Solomon Islands",
    "TRW": "Tarawa Bonriki Kiribati",
    "ROR": "Koror Palau",
    "GUM": "Guam Antonio B. Won Pat",
    "SPN": "Saipan International",
    "PNI": "Pohnpei International",
    # =========================================================================
    # RUSSIA
    # =========================================================================
    "SVO": "Moscow Sheremetyevo",
    "DME": "Moscow Domodedovo",
    "VKO": "Moscow Vnukovo",
    "LED": "St. Petersburg Pulkovo",
    "AER": "Sochi International",
    "KRR": "Krasnodar Pashkovsky",
    "ROV": "Rostov-on-Don Platov",
    "SVX": "Yekaterinburg Koltsovo",
    "KZN": "Kazan International",
    "OVB": "Novosibirsk Tolmachevo",
    "KJA": "Krasnoyarsk Yemelyanovo",
    "IKT": "Irkutsk International",
    "VVO": "Vladivostok International",
    "KHV": "Khabarovsk Novy",
    "UFA": "Ufa International",
    "GOJ": "Nizhny Novgorod Strigino",
    "KGD": "Kaliningrad Khrabrovo",
    "MMK": "Murmansk",
}

# =============================================================================
# AIRLINE CODE MAPPINGS
# =============================================================================
AIRLINE_CODES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "SY": "Sun Country Airlines",
    "AM": "Aeromexico",
    "Y4": "Volaris",
    "VB": "VivaAerobus",
    "BA": "British Airways",
    "AF": "Air France",
    "LH": "Lufthansa",
    "KL": "KLM Royal Dutch Airlines",
    "IB": "Iberia",
    "AZ": "ITA Airways",
    "LX": "Swiss International Air Lines",
    "OS": "Austrian Airlines",
    "SK": "SAS Scandinavian",
    "AY": "Finnair",
    "EI": "Aer Lingus",
    "TP": "TAP Air Portugal",
    "TK": "Turkish Airlines",
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "EY": "Etihad Airways",
    "SV": "Saudi Arabian Airlines",
    "JL": "Japan Airlines",
    "NH": "All Nippon Airways",
    "OZ": "Asiana Airlines",
    "KE": "Korean Air",
    "CX": "Cathay Pacific",
    "SQ": "Singapore Airlines",
    "TG": "Thai Airways",
    "MH": "Malaysia Airlines",
    "GA": "Garuda Indonesia",
    "PR": "Philippine Airlines",
    "CI": "China Airlines",
    "BR": "EVA Air",
    "CA": "Air China",
    "MU": "China Eastern",
    "CZ": "China Southern",
    "QF": "Qantas",
    "NZ": "Air New Zealand",
    "VA": "Virgin Australia",
    "LA": "LATAM Airlines",
    "AV": "Avianca",
    "CM": "Copa Airlines",
    "G3": "Gol Transportes Aereos",
    "AR": "Aerolineas Argentinas",
    "SA": "South African Airways",
    "ET": "Ethiopian Airlines",
    "MS": "EgyptAir",
    "AT": "Royal Air Maroc",
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================
DEFAULT_SCHEDULE_TIME = "02:00"  # Daily collection at 2 AM

# =============================================================================
# UI CONFIGURATION
# =============================================================================
STREAMLIT_PAGE_TITLE = "Flight Price Tracker"
STREAMLIT_PAGE_ICON = "✈️"
STREAMLIT_LAYOUT = "wide"

# Color scheme for visualizations
COLORS = {
    "cheap": "#28a745",      # Green for good prices
    "moderate": "#ffc107",   # Yellow/amber for moderate
    "expensive": "#dc3545",  # Red for expensive
    "primary": "#007bff",    # Blue primary
    "secondary": "#6c757d",  # Gray secondary
    "background": "#f8f9fa", # Light background
}


def validate_iata_code(code: str) -> bool:
    """
    Validate if a given code is a valid IATA airport code.
    
    Args:
        code: The IATA airport code to validate (e.g., 'JFK')
        
    Returns:
        True if the code is valid, False otherwise
    """
    if not code:
        return False
    return code.upper().strip() in VALID_IATA_CODES


def get_airport_name(code: str) -> str:
    """
    Get the full airport name for a given IATA code.
    
    Args:
        code: The IATA airport code (e.g., 'JFK')
        
    Returns:
        The full airport name, or the code itself if not found
    """
    return VALID_IATA_CODES.get(code.upper().strip(), code)


def get_airline_name(code: str) -> str:
    """
    Get the full airline name for a given IATA airline code.
    
    Args:
        code: The IATA airline code (e.g., 'AA')
        
    Returns:
        The full airline name, or the code itself if not found
    """
    return AIRLINE_CODES.get(code.upper().strip(), code)


def get_all_airport_codes() -> list[str]:
    """
    Get a sorted list of all valid IATA airport codes.
    
    Returns:
        List of airport codes sorted alphabetically
    """
    return sorted(VALID_IATA_CODES.keys())


def get_airport_options() -> list[tuple[str, str]]:
    """
    Get airport options formatted for dropdown menus.
    
    Returns:
        List of tuples (code, "CODE - Name") for UI dropdowns
    """
    return [(code, f"{code} - {name}") for code, name in sorted(VALID_IATA_CODES.items())]
