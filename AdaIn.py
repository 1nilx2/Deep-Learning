def AdaIN(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def forward(self, content, style, alpha=1.0):
    assert 0 <= alpha <= 1
    style_feats = self.encode_with_intermediate(style)
    content_feat = self.encode(content)
    t = AdaIN(content_feat, style_feats[-1])
    t = alpha*t + (1 - alpha)*content_feat

    g_t = self.decoder(t)
    g_t_feats = self.encode_with_intermediate(g_t)

    loss_c = self.calc_content_loss(g_t_feats[-1], t)
    loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
    for i in range(1, 4):
        loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
    return loss_c, loss_s