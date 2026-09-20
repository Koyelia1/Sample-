import json
from utility.field_validators import validate_fields

invoice_data = {
    "Invoice Header": {
        "Customer Name": "Publix",
        "Invoice Number": "1812701261",
        "Invoice Date": "2026-02-23",
        "Invoice Total": "365787.26444"
    },
    "Invoice Detail": {
        "Begin Promo  Period": [""],
        "End Promo Period": [""],
        "LINE ITEM TOTAL": [
            "8.00", "24.00", "4.00"
        ],
        "Detail Misc 1": [
            "GRANDS! BIG HONEY"
        ],
        "DESCRIPTION": [
            "20260604 to 20261118 value : 2.00 EL03 EL % Disc / 11190811 / NatVal.oats dark chocolat.grano.bar 210g / Banner FB"
        ],
        "Deal Number": [
            "20260223439471"
        ]
    }
}

result = validate_fields(invoice_data)

print(json.dumps(result, indent=4))