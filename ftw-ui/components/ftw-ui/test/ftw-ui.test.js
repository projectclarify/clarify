import { html, fixture, expect } from '@open-wc/testing';

import '../ftw-ui.js';

describe('FtwUi', () => {
  it('has page "sessions" by default', async () => {
    const el = await fixture(html`
      <ftw-ui></ftw-ui>
    `);

    expect(el.page).to.equal('sessions');
    expect(el.shadowRoot.querySelector('main')).lightDom.to.equal(`
      <view-sessions></view-sessions>
    `);
  });

  it('renders default fallback content', async () => {
    const el = await fixture(html`
      <ftw-ui></ftw-ui>
    `);
    el.page = undefined;

    expect(el.page).to.equal(undefined);
    expect(el.shadowRoot.querySelector('main')).lightDom.to.equal(`
      <view-sessions></view-sessions>
    `);
  });

  it('renders view-account if page property is set to account', async () => {
    const el = await fixture(html`
      <ftw-ui page="account"></ftw-ui>
    `);
    expect(el.shadowRoot.querySelector('main')).lightDom.to.equal(`
      <view-account></view-account>
    `);
  });

  /*
  it('changes the page if a menu link gets clicked', async () => {
    const el = await fixture(html`
      <ftw-ui></ftw-ui>
    `);
    el.shadowRoot.querySelectorAll('header a')[2].click();

    expect(el.page).to.equal('account');
  });

  it('matches the snapshot', async () => {
    const el = await fixture(html`
      <ftw-ui></ftw-ui>
    `);

    expect(el).shadowDom.to.equalSnapshot();
  });

  it('passes the a11y audit', async () => {
    const el = await fixture(html`
      <ftw-ui></ftw-ui>
    `);

    await expect(el).shadowDom.to.be.accessible();
  });

  */
});
