# Replace inline citations with proper LaTeX cite commands

# Gatos dkk. (2006)
s/Gatos dkk\. (2006)/\\textcite{gatos2006}/g

# Pratikakis dkk. (2013)
s/Pratikakis dkk\. (2013)/\\textcite{pratikakis2013}/g

# Pratikakis dkk. (2019) 
s/Pratikakis dkk\. (2019)/\\textcite{pratikakis2013}/g

# Souibgui dkk. (2022) - need context-specific replacement
s/Souibgui dkk\. (2022) mengusulkan Text-DIAE/\\textcite{textdiae2022} mengusulkan Text-DIAE/g
s/Souibgui dkk\. (2022) memperkenalkan DocEnTr/\\textcite{souibgui2022docentr} memperkenalkan DocEnTr/g
s/Souibgui dkk\. (2022) mengidentifikasi/\\textcite{souibgui2022docentr} mengidentifikasi/g
s/Souibgui dkk\. (2022) menunjukkan/\\textcite{souibgui2022docentr} menunjukkan/g

# Souibgui dkk. (2021)
s/Souibgui dkk\. (2021)/\\textcite{erb2021}/g

# Ronneberger dkk. (2015)
s/Ronneberger dkk\. (2015)/\\textcite{ronneberger2015}/g
s/(Ronneberger dkk\., 2015)/(\\cite{ronneberger2015})/g

# Goodfellow dkk. (2014)
s/Goodfellow dkk\. (2014)/\\textcite{goodfellow2014}/g
s/(Goodfellow dkk\., 2014)/(\\cite{goodfellow2014})/g

# Isola dkk. (2017)
s/Isola dkk\. (2017)/\\textcite{isola2017}/g

# Wang dkk. (2004)
s/Wang dkk\. (2004)/\\textcite{wang2004}/g

# Shi dkk. (2016)
s/Shi dkk\. (2016)/\\textcite{shi2016}/g

# Graves dkk. (2006)
s/Graves dkk\. (2006)/\\textcite{graves2006}/g

# Zhu dkk. (2017)
s/Zhu dkk\. (2017)/\\textcite{zhu2017}/g

# Chen dkk. (2018)
s/Chen dkk\. (2018)/\\textcite{chen2018gradnorm}/g

# Johnson dkk. (2016)
s/Johnson dkk\. (2016)/\\textcite{johnson2016}/g

# Zhang dkk. (2024)
s/Zhang dkk\. (2024)/\\textcite{zhang2024}/g

# Kiessling dkk. (2023)
s/Kiessling dkk\. (2023)/\\textcite{kiessling2023}/g

# Martinez dkk. (2024)
s/Martinez dkk\. (2024)/\\textcite{martinez2024}/g

# Thompson dkk. (2023)
s/Thompson dkk\. (2023)/\\textcite{thompson2023}/g

# Zhao dkk. (2019)
s/Zhao dkk\. (2019)/\\textcite{zhao2019}/g

# Ni dkk. (2024)
s/Ni dkk\. (2024)/\\textcite{ni2024}/g

# Jadhav dkk. (2022)
s/Jadhav dkk\. (2022)/\\textcite{jadhav2022}/g

# Diaz dkk. (2021)
s/Diaz dkk\. (2021)/\\textcite{diaz2021}/g

# Baltrusaitis dkk. (2019)
s/Baltrusaitis dkk\. (2019)/\\textcite{baltrusaitis2019}/g

# Radford dkk. (2021)
s/Radford dkk\. (2021)/\\textcite{radford2021}/g

# Vaswani dkk. (2017)
s/Vaswani dkk\. (2017)/\\textcite{vaswani2017}/g

# Vaswani dkk. (2023) - map to 2017 paper
s/Vaswani dkk\. (2023)/\\textcite{vaswani2017}/g

# Raffel dkk. (2020)
s/Raffel dkk\. (2020)/\\textcite{raffel2020}/g

# Raffel dkk. (2023) - map to 2020 paper  
s/Raffel dkk\. (2023)/\\textcite{raffel2020}/g

# Wei dkk. (2022)
s/Wei dkk\. (2022)/\\textcite{wei2022}/g

# Wei dkk. (2024) - map to 2022 paper
s/Wei dkk\. (2024)/\\textcite{wei2022}/g

# Weninger dkk. (2023)
s/Weninger dkk\. (2023)/\\textcite{weninger2023}/g

# Dekel
s/Dekel dkk\./\\textcite{dekel2024}/g

# Wu
s/Wu dkk\./\\textcite{wu2023}/g


# Additional citations
s/Chen dkk\., 2024/\\cite{chen2018gradnorm}/g
s/Zhang dkk\., 2024/\\cite{zhang2024}/g
s/Kießling dkk\., 2023/\\cite{kiessling2023}/g
s/Martínez dkk\., 2024/\\cite{martinez2024}/g
s/Thompson dkk\., 2023/\\cite{thompson2023}/g

# Comment fix
s/Souibgui dkk\. (2022) untuk Text-DIAE/\\textcite{textdiae2022} untuk Text-DIAE/g

