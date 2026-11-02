suppressPackageStartupMessages({library(grid);library(ragg);library(svglite)})

# Original figure's vector drawing primitives and layout, without its sidebar.
c0 <- list(ink="#20242A",muted="#606A74",line="#34495E",
 source="#E9B4C9",source_dark="#A84673",target="#78CBC6",target_dark="#187C79",
 sat="#DDEAFB",sat_dark="#315F9B",lsat="#E8E0F5",lsat_dark="#68489B",
 test="#FFF0C2",test_dark="#8A6500",result="#E7F1E2",result_dark="#3E6C35")
rr <- function(x,y,w,h,fill="white",edge=c0$line,lwd=1.4,r=.012)
 grid.roundrect(x,y,w,h,r=unit(r,"npc"),gp=gpar(fill=fill,col=edge,lwd=lwd))
tx <- function(s,x,y,size=11,color=c0$ink,bold=FALSE,just="centre",italic=FALSE)
 grid.text(s,x,y,just=just,gp=gpar(col=color,fontsize=size,
  fontface=if(bold)"bold" else if(italic)"italic" else "plain",fontfamily="Helvetica"))
mt <- function(e,x,y,size=11,color=c0$ink)
 grid.text(e,x,y,gp=gpar(col=color,fontsize=size))
ar <- function(x0,y0,x1,y1,color=c0$line,lwd=2)
 grid.lines(unit(c(x0,x1),"npc"),unit(c(y0,y1),"npc"),
  arrow=arrow(type="closed",length=unit(.12,"inches")),gp=gpar(col=color,lwd=lwd,lineend="round"))
people <- function(x,y,fill,scale=1) {
 dx<-c(-.030,-.010,.010,.030,-.020,0,.020)*scale
 dy<-c(0,.005,.005,0,-.027,-.024,-.027)*scale
 for(i in seq_along(dx)) {
  grid.circle(x+dx[i],y+dy[i],r=.006*scale,gp=gpar(fill=fill,col=c0$ink,lwd=.65))
  grid.rect(x+dx[i],y+dy[i]-.015*scale,width=.010*scale,height=.014*scale,
   gp=gpar(fill=fill,col=c0$ink,lwd=.65))
 }
}

draw <- function() {
 grid.newpage()
 tx('Fig 1. Binary and Cox layer-specific source-anchored transfer learning',.5,.975,size=18,bold=TRUE)
 tx('Older source age cutoffs: 60 / 55 / 50 years; target <50 years | 70% target-training, 30% target-test',
    .5,.944,size=10.5,color=c0$muted)
 for(j in 1:2) {
  center<-c(.32,.68)[j]; color<-c(c0$sat_dark,c0$lsat_dark)[j]
  fill<-c(c0$sat,c0$lsat)[j]
  rr(center,.850,.30,.070,fill,color,r=.006)
  tx(c('Binary LSAT-EN','LSAT-Cox-EN')[j],center,.867,size=16,color=color,bold=TRUE)
  tx(c('Horizon-specific logistic likelihood','Cox partial likelihood')[j],center,.835,size=9.3,color=color)
  layer_names<-c('mRNA','miRNA','Methylation','Protein','Mutation','CNV')
  layer_fills<-c('#F7C9A9','#F9E58B','#BDE3DB','#D5E2F1','#E8D7E8','#DFE9CF')
  layer_x<-center+c(-.1125,-.0375,.0375,.1125,-.0375,.0375)
  layer_y<-c(rep(.790,4),rep(.748,2))
  for(i in 1:6) {
   rr(layer_x[i],layer_y[i],.065,.034,layer_fills[i],c0$line,lwd=.8,r=.003)
   tx(layer_names[i],layer_x[i],layer_y[i],size=6.6,bold=TRUE)
   if(i>=5)ar(layer_x[i],.728,center,.707,color,.8)
  }
  people(center-.09,.670,c0$source,.95)
  tx('Older source',center+.04,.683,size=11,color=c0$source_dark,bold=TRUE)
  tx(c('Fit source-only logistic EN','Fit source-only Cox EN')[j],center+.04,.656,size=8.5,color=c0$muted)
  if(j==1)mt(expression(hat(beta)[S*g]^bin),center+.04,.625,size=13,color=c0$source_dark)
  else mt(expression(hat(beta)[S*g]^Cox),center+.04,.625,size=13,color=c0$source_dark)
  ar(center,.600,center,.550,color,1.8)
  tx('Layer-weighted transfer',center,.577,size=8.5,color=color,italic=TRUE)
  rr(center,.475,.30,.110,'#F7FBFB',c0$target_dark,r=.006)
  people(center-.085,.500,c0$target,.8)
  tx('Source + 70% target-training',center+.025,.512,size=9.0,color=c0$target_dark,bold=TRUE)
  tx('Inner 5-fold CV | source via fixed anchor',center+.025,.482,size=7.5)
  mt(expression(paste('Tune ',lambda,', ',alpha,', and each ',tau[g])),center+.02,.452,size=9.3,color=color)
  rr(center,.355,.32,.072,'white',color,lwd=1,r=.005)
  if(j==1)mt(expression(-ell[T]^bin*group('(',beta,')')+P[EN](beta)+
   frac(1,2)*sum(tau[g]*group('||',beta[g]-hat(beta)[S*g]^bin,'||')[2]^2,g)),center,.355,size=8.1)
  else mt(expression(-ell[T]^Cox*group('(',beta,')')+P[EN](beta)+
   frac(1,2)*sum(tau[g]*group('||',beta[g]-hat(beta)[S*g]^Cox,'||')[2]^2,g)),center,.355,size=8.1)
  ar(center,.315,center,.280,color,1.6)
  rr(center,.240,.28,.070,c0$test,c0$test_dark,r=.006)
  tx('Refit on all 70% target-training',center,.256,size=9.3,bold=TRUE)
  tx('Fixed source anchor; test on untouched 30%',center,.229,size=8.0,color=c0$test_dark)
 }
 ar(.32,.202,.43,.160,c0$sat_dark,1.8)
 ar(.68,.202,.57,.160,c0$lsat_dark,1.8)
 rr(.5,.107,.76,.085,c0$result,c0$result_dark,lwd=1.4,r=.006)
 tx('Target-cohort performance across 10 repeated splits',.5,.129,size=13,color=c0$result_dark,bold=TRUE)
 tx('Binary: 2-4-year AUC (SE) | Cox: Harrell C-statistic and 2-5-year IPCW AUC (SE)',.5,.100,size=9.5)
 mt(expression(P[EN](beta)==lambda*(alpha*group('||',beta,'||')[1]+
  frac(1-alpha,2)*group('||',beta,'||')[2]^2)),.5,.037,size=10)
 tx('Target-only likelihood; layer-specific borrowing toward source anchors; EN sparsity for biomarker prioritization',
  .5,.010,size=8.5,color=c0$muted)
}
for(ext in c('png','svg')) {
 path<-file.path('assets',paste0('lsat-workflow.',ext))
 if(ext=='png')agg_png(path,width=7200,height=6000,res=600,background='white')
 else svglite(path,width=12,height=10,bg='white')
 draw();dev.off()
}
