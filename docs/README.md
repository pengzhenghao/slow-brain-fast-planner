# Project webpage

Static HTML and CSS with figures from the paper. The paper is linked on [arXiv](https://arxiv.org/abs/2606.20458).

## Local preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8000/. No build step is required.

## GitHub Pages

1. In the public repository, set **Settings → Pages → Source** to **GitHub Actions**.
2. Run **Actions → Deploy project page → Run workflow** on `main`.
3. Open https://pengzhenghao.github.io/slow-brain-fast-planner/.

Deployment is manual; changing repository visibility alone does not deploy the site.

## Figures

The page uses figures 1, 3, 5 and 7 from the [public arXiv version](https://arxiv.org/html/2606.20458v1),
converted from PNG to WebP at quality 94 without cropping or changing their content:

- `assets/teaser.webp` ← `teaser.png`
- `assets/method.webp` ← `corl_method.png`
- `assets/qualitative.webp` ← `quali.png`
- `assets/real-world.webp` ← `real_world_robot.png`

The paper is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The single-column layout follows the traditional [MetaDriverse project-page style](https://metadriverse.github.io/scenestreamer/).
