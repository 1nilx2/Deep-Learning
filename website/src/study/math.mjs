export function softmax(logits) {
  const peak = Math.max(...logits);
  const exps = logits.map(value => Math.exp(value - peak));
  const sum = exps.reduce((a,b) => a+b,0);
  return exps.map(value => value/sum);
}
export function weightedCE(logits, target, weight) {
  const peak=Math.max(...logits);
  const logNormalizer=peak+Math.log(logits.reduce((sum,value)=>sum+Math.exp(value-peak),0));
  return weight*(logNormalizer-logits[target]);
}
export function dpoLoss(chosenRatio, rejectedRatio, beta) {
  const x=-beta*(chosenRatio-rejectedRatio);
  return Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)));
}
export function bigramMatch(response, captions) {
  const grams = value => {
    const words=value.toLowerCase().match(/[a-z0-9]+/g)||[];
    return new Set(words.slice(1).map((word,i)=>words[i]+' '+word));
  };
  const query=grams(response);
  const scores=captions.map(caption=>[...grams(caption)].filter(gram=>query.has(gram)).length);
  const best=Math.max(...scores);
  const winners=scores.flatMap((score,i)=>score===best?[i]:[]);
  return {scores, index:best>0&&winners.length===1?winners[0]:null};
}
