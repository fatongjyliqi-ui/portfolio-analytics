import math

# Funktionen

def mittelwert(liste):
    return sum(liste) / len(liste)

def standardabweichung(liste):
    m = mittelwert(liste)
    abw_q = [(x - m)**2 for x in liste]
    return math.sqrt(sum(abw_q) / (len(liste) - 1))

def annualisieren_rendite(tagesrendite, tage=252):
    return tagesrendite * tage

def annualisieren_vol(tagesvol, tage=252):
    return tagesvol * math.sqrt(tage)

def sharpe(rendite, vol, rf=0):
    return (rendite - rf) / vol

def klassifiziere_volatilitaet(annual_vol):
    if annual_vol < 0.10:
        return "NIEDRIG (z.B. Anleihen, Geldmarkt)"
    elif annual_vol < 0.20:
        return "MITTEL (z.B. breit gestreute Aktien)"
    elif annual_vol < 0.35:
        return "HOCH (z.B. Einzelaktien, Schwellenländer)"
    else:
        return "SEHR HOCH (z.B. Krypto, gehebelte Produkte)"
    
def drawdown(renditen):
    kum_rendite = [1]
    for x in range(len(renditen)):
        kum_rendite.append(kum_rendite[-1] * (1 + renditen[x]))
    peak = 0
    dd = 0
    for value in kum_rendite:
        if value > peak:
            peak = value
        current_dd = (peak - value) / peak if peak != 0 else 0
        if current_dd > dd:
            dd = current_dd
    return dd

def report(renditen, asset_name="Asset"):
        n = len(renditen)
        m = mittelwert(renditen)
        s = standardabweichung(renditen)
        ar = annualisieren_rendite(m)
        av = annualisieren_vol(s)
        sr = sharpe(ar, av)
        klasse = klassifiziere_volatilitaet(av)
        positive = len([r for r in renditen if r > 0])
        negative = len([r for r in renditen if r < 0])
        dd = drawdown(renditen)
        print("=" * 50)
        print(f"  RENDITEN-ANALYSE: {asset_name}")
        print("=" * 50)
        print(f"Anzahl Renditen: {n}")
        print(f"Beste Tagesrendite: {max(renditen):7.2f}%")
        print(f"Schlechteste Tagesrendite: {min(renditen):7.2f}%")
        print("-" * 50)
        print(f"Mittelwert: {m:7.2f}%")
        print(f"Standardabweichung: {s:7.2f}%")
        print("-" * 50)
        print(f"Annualisierte Rendite: {ar:7.2f}%")
        print(f"Annualisierte Volatilität: {av:7.2f}%")
        print(f"Sharpe-Ratio: {sr:7.2f}")
        print("-" * 50)
        print(f"Volatilitätsklasse: {klasse}")
        print("-" * 50)
        print(f"Drawdown: {dd*100:7.2f}%")
        print("=" * 50)

beispiel_renditen = [
    0.012, -0.005, 0.018, 0.003, -0.022,
    0.015, 0.008, -0.011, 0.025, -0.018,
    0.030, -0.005, 0.012, 0.007, -0.014,
    0.022, 0.005, -0.008, 0.018, 0.011,
    -0.003, 0.014, 0.021, -0.009, 0.007,
    0.013, -0.017, 0.024, 0.006, -0.012,
    0.011, 0.018, -0.020, 0.015, 0.009,
    -0.006, 0.022, 0.013, -0.014, 0.008,
    0.017, 0.005, -0.011, 0.019, 0.010,
    -0.007, 0.014, 0.022, -0.013, 0.006,
]
report(beispiel_renditen, asset_name="Beispiel-Asset")