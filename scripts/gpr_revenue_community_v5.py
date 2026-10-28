"""Collect net revenue using current Community Vnstock rights; never bypass caps."""
import contextlib,io,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import vnstock

OUT=Path('output/gpr_revenue_community');OUT.mkdir(parents=True,exist_ok=True)
KEY=os.environ.get('VNSTOCK_API_KEY','')
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def main():
 meta={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'source':'vci','requested_period':'quarter','target_years':[2018,2025],'scope':'actual source statement scope; verify against original filings before model use','status':'starting','errors':[]}
 save('summary.json',meta)
 if not KEY:
  meta.update(status='blocked',reason='VNSTOCK_API_KEY secret missing; Community login required');save('summary.json',meta);return 2
 try:
  # Do not persist authentication messages or account identifiers.
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):vnstock.login(KEY)
  meta['tier']=vnstock.status().get('tier','unknown')
 except Exception as e:
  meta.update(status='blocked',reason=str(e).replace(KEY,'[REDACTED]')[:350],error_code=str(getattr(e,'code','unknown')),exception_type=type(e).__name__);save('summary.json',meta);return 2
 if meta['tier']=='guest':
  meta.update(status='blocked',reason='Community authentication was not established');save('summary.json',meta);return 2
 symbols=pd.read_csv('data/gpr_revenue_symbols.csv').symbol.tolist()
 symbols=sorted(symbols,key=lambda s:(s!='HPG',s))
 errors=[];rows=[];metadata=[];completed=0
 for idx,symbol in enumerate(symbols,1):
  path=OUT/(symbol+'_raw.csv');ap=OUT/(symbol+'_metadata.json')
  if path.exists() and ap.exists():
   frame=pd.read_csv(path);attrs=json.loads(ap.read_text())
  else:
   try:
    frame=vnstock.Fundamental().equity(symbol).income_statement(period='quarter',format='long',source='vci')
    if frame is None or frame.empty:raise ValueError('empty_report')
    required={'period','id','unit','value'}
    if not required.issubset(frame.columns):raise ValueError('unexpected_schema')
    attrs={k:v for k,v in frame.attrs.items() if k.startswith(('tier_cap_','cap_')) or k=='source'}
    frame.to_csv(path,index=False);save(ap.name,attrs)
   except Exception as e:
    msg=str(e).replace(KEY,'[REDACTED]')[:500]
    errors.append({'symbol':symbol,'type':type(e).__name__,'message':msg})
    print(idx,len(symbols),symbol,'ERROR',type(e).__name__,flush=True)
    # Stop on rights/auth/account quota errors. No repeated calls or source bypass.
    if any(k in (type(e).__name__+' '+msg).lower() for k in ['entitlement','auth','hour_limit','day_limit','month_limit','device','license']):break
    time.sleep(8);continue
   time.sleep(8)
  q=frame[frame.id.eq('IS_NET_REVENUE')].copy()
  q['symbol']=symbol;q['period_q']=q.period.astype(str).str.replace('-Q','Q',regex=False)
  q['revenue_raw']=pd.to_numeric(q.value,errors='coerce');q['source']='vci'
  # Preserve reported units and source values; no unverified cumulative conversion.
  q['flow_basis']='as_returned_not_yet_reconciled_to_annual'
  if q.duplicated(['symbol','period_q']).any():
   errors.append({'symbol':symbol,'type':'DuplicateRevenue','message':'Multiple net revenue rows; excluded from merge'})
  else:rows.append(q[['symbol','period_q','revenue_raw','unit','source','flow_basis']])
  completed+=1;metadata.append({'symbol':symbol,'periods':sorted(frame.period.astype(str).unique()),'caps':attrs,'revenue_rows':len(q)})
  pd.DataFrame(metadata).to_json(OUT/'coverage.json',orient='records',force_ascii=False,indent=2)
  print(idx,len(symbols),symbol,'periods',frame.period.nunique(),'revenue_rows',len(q),flush=True)
  if rows:
   panel=pd.concat(rows,ignore_index=True);panel.to_csv(OUT/'revenue_all_returned.csv',index=False)
   target=panel[panel.period_q.between('2018Q1','2025Q4')].copy();target.to_csv(OUT/'revenue_target_2018_2025.csv',index=False)
  meta.update(status='running',symbols_requested=len(symbols),symbols_completed=completed,errors=errors)
  save('summary.json',meta)
 meta.update(status='completed' if completed==len(symbols) else 'partial_or_blocked',symbols_requested=len(symbols),symbols_completed=completed,errors=errors)
 meta['historical_coverage_warning']='Community rights may retain only 8 recent quarters. This collection does not establish complete 2018–2025 revenue coverage.'
 save('summary.json',meta)
 return 0 if completed else 2
if __name__=='__main__':raise SystemExit(main())
