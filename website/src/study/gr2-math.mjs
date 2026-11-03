// Section 4.3, equations (14)–(17). Inputs are valid, one-based target ranks.
export function rankReward(before,after,n=10,format=true,alpha=.1,conditional=true){
  if(!Number.isInteger(n)||n<1||![before,after].every(x=>Number.isInteger(x)&&x>=1&&x<=n))throw new RangeError('Ranks must lie in the candidate list.');
  const rank=(before-after)/n;
  const bonus=(!conditional||rank>0||before===1)&&format?alpha:0;
  return {rank,bonus,total:rank+bonus};
}
export function groupAdvantages(rewards){
  const mean=rewards.reduce((a,b)=>a+b,0)/rewards.length;
  const std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/rewards.length);
  return {mean,std,advantages:rewards.map(r=>std===0?0:(r-mean)/std)};
}
