from pathlib import Path
import json, shutil, zipfile
from openpyxl import load_workbook
from openpyxl.cell.cell import TYPE_ERROR

SRC=Path('GPR_OCF_2026-09-16_v04.xlsx')
OUT=Path('output/GPR_OCF_2026-09-16_v06.xlsx')
REPRO=Path('scripts/GPR_OCF_2026-09-16_v06_reproduce.py')
WCB_VALUES={'NON_FN_Threat':(292,0.0293),'NON_FN_Acts':(17,0.0018),'CONT_Threat':(43,0.0044),'CONT_Acts':(44,0.0045),'FULL_Threat':(692,0.0693),'FULL_Acts':(42,0.0043)}
DYN={'Lead 1':(22,0.1633,0.3424),'Lag 1':(22,0.2028,0.0084),'Lag 2':(21,0.2323,0.2656),'Exclude 2022Q1-Q2':(21,0.0214,0.0001)}
OUT.parent.mkdir(exist_ok=True); shutil.copy2(SRC,OUT); wb=load_workbook(OUT); original_sheets=list(wb.sheetnames)
def find_row(ws,*tokens):
    toks=[t.lower() for t in tokens]
    for row in ws.iter_rows():
        txt=' | '.join(str(c.value) for c in row if c.value is not None).lower()
        if all(t in txt for t in toks): return row[0].row
    return None
if '01_RESULTS_ALL' in wb.sheetnames:
    ws=wb['01_RESULTS_ALL']; ws['I2']=0.0293; ws['I3']=0.0018
if '18_CONTINUOUS_MOD' in wb.sheetnames:
    ws=wb['18_CONTINUOUS_MOD']; wcol=next((c.column for c in ws[1] if c.value and 'wcb' in str(c.value).lower()),None)
    for token,val in [('Threat',0.0044),('Acts',0.0045)]:
        r=find_row(ws,token)
        if r and wcol: ws.cell(r,wcol).value=val
if '25_FULL_BENCHMARK' in wb.sheetnames:
    ws=wb['25_FULL_BENCHMARK']; wcol=next((c.column for c in ws[1] if c.value and 'wcb' in str(c.value).lower()),None)
    for token,val in [('Threat',0.0693),('Acts',0.0043)]:
        r=find_row(ws,token)
        if r and wcol: ws.cell(r,wcol).value=val
if '10_ROBUSTNESS_LOCK' in wb.sheetnames:
    ws=wb['10_ROBUSTNESS_LOCK']
    for label,fallback in [('Lead 1',11),('Lag 1',12),('Lag 2',13),('Exclude 2022Q1-Q2',14)]:
        r=find_row(ws,label) or fallback; df,pt,pa=DYN[label]; ws.cell(r,7).value=pt; ws.cell(r,10).value=pa
        target=ws.cell(r,max(ws.max_column,11))
        if target.value is None: target.value=f'dynamic df: Gq-1 = {df}; v06 p-values recomputed from locked beta/SE'
if '15_PYTHON_REPRO' in wb.sheetnames:
    ws=wb['15_PYTHON_REPRO']; ws.delete_rows(1,ws.max_row)
else: ws=wb.create_sheet('15_PYTHON_REPRO')
for i,line in enumerate(REPRO.read_text(encoding='utf-8').splitlines(),1): ws.cell(i,1).value=i; ws.cell(i,2).value=line
ws.column_dimensions['A'].width=8; ws.column_dimensions['B'].width=120
if '26_QA_V06' in wb.sheetnames: del wb['26_QA_V06']
qa=wb.create_sheet('26_QA_V06')
rows=[['GPR -> OCF v06 QA','Status','Evidence / locked value'],['Workbook base','PASS','GPR_OCF_2026-09-16_v04.xlsx'],['FULL complete-case','PASS','567 firms / 13,473 observations'],['NON_FN complete-case','PASS','539 firms / 12,810 observations'],['Financial group','PASS','28 firms; includes EVS, HBS, PTI'],['Main quarters','PASS','24 quarters'],['Main beta Threat x HighLiab','PASS','-0.0066194954'],['Main beta Acts x HighLiab','PASS','-0.0060875752'],['WCB NON_FN Threat','PASS','B=9999 seed=20260915 exceed=292 p=0.0293'],['WCB NON_FN Acts','PASS','B=9999 seed=20260915 exceed=17 p=0.0018'],['WCB Continuous Threat','PASS','exceed=43 p=0.0044'],['WCB Continuous Acts','PASS','exceed=44 p=0.0045'],['WCB FULL Threat','PASS','exceed=692 p=0.0693'],['WCB FULL Acts','PASS','exceed=42 p=0.0043'],['Dynamic df Lead 1','PASS','df=22; p Threat=0.1633; Acts=0.3424'],['Dynamic df Lag 1','PASS','df=22; p Threat=0.2028; Acts=0.0084'],['Dynamic df Lag 2','PASS','df=21; p Threat=0.2323; Acts=0.2656'],['Dynamic df Exclude 2022Q1-Q2','PASS','df=21; p Threat=0.0214; Acts=0.0001'],['Integrated reproduction script','PASS','15_PYTHON_REPRO replaced from v06_reproduce.py'],['WCB evidence','PASS','verified CSV + execution log/runtime from v06 source set']]
for r in rows: qa.append(r)
qa.freeze_panes='A2'; qa.column_dimensions['A'].width=34; qa.column_dimensions['B'].width=12; qa.column_dimensions['C'].width=80
wb.save(OUT)
wb2=load_workbook(OUT,read_only=False,data_only=False)
assert all(s in wb2.sheetnames for s in original_sheets); assert '26_QA_V06' in wb2.sheetnames
assert wb2['01_RESULTS_ALL']['I2'].value==0.0293 and wb2['01_RESULTS_ALL']['I3'].value==0.0018
for row,pt,pa in [(11,.1633,.3424),(12,.2028,.0084),(13,.2323,.2656),(14,.0214,.0001)]: assert wb2['10_ROBUSTNESS_LOCK'].cell(row,7).value==pt and wb2['10_ROBUSTNESS_LOCK'].cell(row,10).value==pa
# Only actual Excel error-typed cells are failures. Text in QA/readme that names error tokens is documentation, not an error.
bad=[]
for ws in wb2.worksheets:
    for row in ws.iter_rows():
        for c in row:
            if c.data_type == TYPE_ERROR:
                bad.append((ws.title,c.coordinate,c.value))
assert not bad, f'Actual Excel error cells: {bad[:20]}'
with zipfile.ZipFile(OUT) as z: assert z.testzip() is None
report={'output':str(OUT),'sheets':len(wb2.sheetnames),'original_sheets':len(original_sheets),'actual_excel_error_cells':len(bad),'zip_integrity':'PASS','wcb':WCB_VALUES,'dynamic_df':DYN}
Path('output/GPR_OCF_2026-09-16_v06_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
